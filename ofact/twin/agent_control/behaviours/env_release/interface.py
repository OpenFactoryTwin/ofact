"""
Build the interface between the negotiation and the real process_execution in the simulation/ shop_floor etc.
Communication to the env
@last update: 21.08.2023
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour
from ofact.twin.agent_control.helpers.debug_str import get_debug_str
from ofact.twin.agent_control.helpers.sort_process_executions import _get_sorted_process_executions
from ofact.twin.change_handler.reactions import ProcessExecutionAction, Action

# Imports Part 2: PIP Imports
if TYPE_CHECKING:
    from datetime import timedelta
    from ofact.twin.change_handler.change_handler import ChangeHandler
    from ofact.twin.state_model.processes import ProcessExecution
    from ofact.twin.state_model.sales import Order


class EnvInterfaceBehaviour(DigitalTwinCyclicBehaviour):
    """
    The env interface is used as interface between the agent and the environment, respectively the change handler
    From the change_handler, process_executions are forwarded (if subscribed before). Subsequently, the planning
    is triggered when changes with relevant effects are available.
    On the way from the agent to the change handler, process_executions are forwarded or a notice
    that no action would be taken is passed.
    """

    def __init__(self):
        super(EnvInterfaceBehaviour, self).__init__()

    def inform_actual_process_execution(self, actual_process_execution: ProcessExecution):
        """called by the change_handler to inform about a subscribed process_execution"""
        # print(self.agent.name, "INFORM")
        # if "order_agent" in self.agent.name:
        #     print(self.agent.name, "process finished", actual_process_execution.process.name,
        #           actual_process_execution.connected_process_execution in self.agent.process_executions_in_execution)
        if actual_process_execution.connected_process_execution in self.agent.process_executions_in_execution:
            self.agent.process_executions_in_execution.pop(actual_process_execution.connected_process_execution)
            self.agent.process_executions_finished.append(actual_process_execution.connected_process_execution)
            # ToDo: intern handling
        else:
            pass  # ToDo Case: resource (not main_resource) is subscribed on pe but did not track it in his lists

class OrderPoolEnvInterfaceBehaviour(EnvInterfaceBehaviour):

    def __init__(self):
        """
        reaction_expected: a bool that specifies if the change_handler expects an action
        """
        super(OrderPoolEnvInterfaceBehaviour, self).__init__()
        self.reaction_expected: bool = False

        self.current_round = 0  # strongly dependent on the counter of the agents model (find an elegant way)

    def plan_order_release(self, order: Order):
        self.agent.plan_order_release(order, agent_name=self.agent.name, env_interface_behaviour=self,
                                      order_subscription= True)

    def reminder_for_order_release(self, order: Order):

        self.agent.reminder_for_order_release(order)

    async def on_end(self):
        await super().on_end()


class OrderEnvInterfaceBehaviour(EnvInterfaceBehaviour):

    def __init__(self):
        """
        reaction_expected: a bool that specifies if the change_handler expects an action
        """
        super(OrderEnvInterfaceBehaviour, self).__init__()
        self.reaction_expected: bool = False

        self.current_round = 0  # strongly dependent on the counter of the agents model (find an elegant way)

    async def run(self):
        await super().run()

        if self.agent.process_executions_queue:
            self.subscribe_on_ape()

        if self.reaction_expected:
            # if str(datetime.now().microsecond)[0:3] == "100":
            #     print(f"{datetime.now()} | [{self.agent.name:35}] Reaction expected")
            if self.agent.activity != self.agent.Activity.NO_ACTION:
                await self.agent.waiting_for_planning_end

            self.reaction_expected = False
            self.current_round += 1
            # from datetime import datetime
            await self.agent.change_handler.go_on(agent_name=self.agent.name,
                                                  round_=self.current_round)

    def subscribe_on_ape(self):
        process_executions = []
        while self.agent.process_executions_queue:
            self_agent_name, planned_process_execution = self.agent.process_executions_queue.pop(0)

            self.agent.change_handler: ChangeHandler
            self.agent.change_handler.subscribe_on_ape(planned_process_execution=planned_process_execution,
                                                       agent_name=self_agent_name,
                                                       env_interface_behaviour=self)

            self.agent.process_executions_in_execution[planned_process_execution] = self_agent_name
            process_executions.append(planned_process_execution)

        self.agent.process_executions_order.extend(process_executions)
        self.agent.process_executions_order = _get_sorted_process_executions(self.agent.process_executions_order)

    def inform_actual_process_execution(self, actual_process_execution: ProcessExecution):

        super().inform_actual_process_execution(actual_process_execution=actual_process_execution)
        self.agent.process_executions_order.remove(actual_process_execution.connected_process_execution)
        self.agent.process_executions_order.append(actual_process_execution)
        self.agent.process_executions_order = _get_sorted_process_executions(self.agent.process_executions_order)

    def request_go_on(self):
        """Called from the change_handler to trigger a reaction to go_on"""
        self.reaction_expected = True
        self.agent.waiting_for_planning_end = asyncio.get_event_loop().create_future()
        self.agent.activity = self.agent.Activity.IN_PLANNING
        if self.agent.waiting_on_next_round is not None:
            try:
                self.agent.waiting_on_next_round.set_result(True)
            except asyncio.exceptions.InvalidStateError:
                pass
            self.agent.waiting_on_next_round = None
        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Request go on: {self.agent.activity}")

    async def on_end(self):
        await super().on_end()


class ResourceEnvInterfaceBehaviour(EnvInterfaceBehaviour):

    def __init__(self):
        """
        process_executions_to_forward: a list of process_executions_components that should be forwarded to another agent
        """
        super(ResourceEnvInterfaceBehaviour, self).__init__()

        release_template = {"metadata": {"performative": "inform",
                                         "ontology": "Release",
                                         "language": "OWL-S"}}
        cancel_template = {"metadata": {"performative": "inform",
                                        "ontology": "Cancel",
                                        "language": "OWL-S"}}
        self.templates = [release_template, cancel_template]

        self.metadata_conditions_lst = [{"performative": "inform", "ontology": "Release"},
                                        {"performative": "inform", "ontology": "Cancel"}]

        self.pre_reaction = False

        self.released_process_executions = []
        self.task_released_process_executions = None
        self.rejected_process_executions = []
        self.rejected_process_executions2 = []
        self.task_rejected_process_executions = None

        self.tasks = []

    def initial_subscription(self):
        """initial subscription on the process_executions_components
        that include the resources managed by the ResourceAgent"""
        for resource in self.agent._resources:
            self.agent.change_handler.subscribe_on_entity(entity=resource,
                                                          agent_name=self.agent.name,
                                                          env_interface_behaviour=self)

    async def run(self):
        await super().run()
        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" On hold:",
        #       [process_execution.identification
        #        for process_execution, _ in self.agent.process_executions_on_hold.items()])
        msg_process_execution_release_or_rejection_received = \
            await self.agent.receive_msg(self, timeout=10, metadata_conditions_lst=self.metadata_conditions_lst)
        # print(get_debug_str(self.agent.name, self.__class__.__name__),
        #       msg_process_execution_release_or_rejection_received)
        if msg_process_execution_release_or_rejection_received is not None:
            msg_content, msg_sender, msg_ontology, msg_performative = \
                msg_process_execution_release_or_rejection_received
            handle_received_msg_task = asyncio.create_task(self._handle_received_msg(msg_content, msg_ontology))

    async def _handle_received_msg(self, msg_content, msg_ontology):
        """Receive messages and perform the consequences of messages received"""
        process_executions = msg_content

        if msg_ontology == "Release":
            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       f" Release ProcessExecutions")  # : {list(map(lambda obj: obj.identification, process_executions))}")
            self.released_process_executions.extend(process_executions)
            if self.task_released_process_executions is None:
                self.task_released_process_executions = asyncio.create_task(self._release_process_executions())
                self.tasks.append(self.task_released_process_executions)
            elif self.task_released_process_executions.done():
                self.task_released_process_executions = asyncio.create_task(self._release_process_executions())
                self.tasks.append(self.task_released_process_executions)

        elif msg_ontology == "Cancel":
            self.rejected_process_executions.extend(process_executions)
            self.rejected_process_executions2.extend(process_executions)
            if self.task_rejected_process_executions is None:
                self.task_rejected_process_executions = asyncio.create_task(self._handle_rejected_process_executions())
                self.tasks.append(self.task_rejected_process_executions)
            elif self.task_rejected_process_executions.done():
                self.task_rejected_process_executions = asyncio.create_task(self._handle_rejected_process_executions())
                self.tasks.append(self.task_rejected_process_executions)

        # print(get_debug_str(self.agent.name, self.__class__.__name__),
        # "PE on Hold:", len(self.agent.process_executions_on_hold), self.agent.reaction_expected)

        if self.agent.reaction_expected:
            self.reaction_expected = False
            tasks, self.tasks = self.tasks, []
            for task in tasks:
                await task

            if self.agent.process_executions_on_hold:
                return

            await self.agent.change_handler.go_on(agent_name=self.agent.name)

        else:
            self.pre_reaction = True

        # abc = [(pe.identification, pe.process.name, pe.order.identification, name)
        #        for pe, name in self.agent.process_executions_on_hold.items()]
        # print(self.agent.name, "Process Execution on Hold: ", abc)
        # print([(process_execution.identification, process_execution.process.name,
        #         process_execution.order.identification)
        #        for process_execution in self.released_process_executions])

        # print(get_debug_str(self.agent.name, self.__class__.__name__), self.agent.process_executions_on_hold)

    def request_go_on(self):
        """Called from the change_handler to trigger a reaction to go_on"""
        if self.pre_reaction:
            # print("Pre reaction:", self.agent.name)

            task = asyncio.create_task(self.agent.change_handler.go_on(agent_name=self.agent.name))
            self.pre_reaction = False
            return

        self.agent.reaction_expected = True
        # print(f"{datetime.now()} | [{self.agent.name:35}] Request go on")

    async def _release_process_executions(self):
        """release the process_executions_components to the environment for execution"""
        while self.released_process_executions:
            # print(self.agent.name, len(self.agent.process_executions_on_hold))
            released_process_execution_dict = self.released_process_executions.pop(0)
            released_process_execution = released_process_execution_dict["ProcessExecution"]

            if released_process_execution not in self.agent.process_executions_on_hold:
                await self._wait_for_process_execution(released_process_execution)
                # print(get_debug_str(self.agent.name, self.__class__.__name__) + "Loop end")

            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       f" Release PE {released_process_execution.identification}")
            requester_agent_name = self.agent.process_executions_on_hold[released_process_execution]
            deviation_tolerance_time_delta_order = released_process_execution_dict["Deviation Tolerance Time Delta"]
            notification_time_delta_order = released_process_execution_dict["Notification Time Delta"]

            self._execute_process(requester_agent_name=requester_agent_name,
                                  planned_process_execution=released_process_execution,
                                  deviation_tolerance_time_delta_order=deviation_tolerance_time_delta_order,
                                  notification_time_delta_order=notification_time_delta_order)

            del self.agent.process_executions_on_hold[released_process_execution]
        # print(self.agent.name, len(self.agent.process_executions_on_hold))

    def _execute_process(self, requester_agent_name, planned_process_execution,
                         deviation_tolerance_time_delta_order: Optional[timedelta] = None,
                         notification_time_delta_order: Optional[timedelta] = None):
        """Pass the planned process_execution to the external objects/
        append the process_executions_components to the change_handler"""

        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" Execute ProcessExecution {planned_process_execution.process.name} with ID "
        #       f"{planned_process_execution.identification} from requester {requester_agent_name}")

        deviation_tolerance_time_delta = min(self.agent.deviation_tolerance_time_delta,
                                             deviation_tolerance_time_delta_order)  # ToDo: None problem
        notification_time_delta = min(self.agent.notification_time_delta,
                                      notification_time_delta_order)

        # pass the process_execution to the digital_twin
        reaction = (
            ProcessExecutionAction(type_=Action.Types.PROCESS_EXECUTION, agent_name=self.agent.name,
                                   planned_process_execution=planned_process_execution, env_interface_behaviour=self,
                                   ape_subscription=True, deviation_tolerance_time_delta=deviation_tolerance_time_delta,
                                   notification_time_delta=notification_time_delta))
        self.agent.change_handler.act(action=reaction)

        self.agent.process_executions_in_execution[planned_process_execution] = requester_agent_name

    async def _handle_rejected_process_executions(self):
        """reset the actions taken for the planning of the rejected process_executions_components before"""

        while self.rejected_process_executions:
            rejected_process_execution = self.rejected_process_executions.pop(0)
            if rejected_process_execution.main_resource in self.agent.resources:
                if rejected_process_execution not in self.agent.process_executions_on_hold:
                    await self._wait_for_process_execution(rejected_process_execution)

        rejected_process_executions = self.rejected_process_executions2.copy()
        self.rejected_process_executions2 = []

        # print(get_debug_str(self.agent.name, self.__class__.__name__), [pe.process.identification
        #                                                                 for pe in rejected_process_executions])

        self.agent.reset_planned_process_executions(unused_process_executions=rejected_process_executions,
                                                    blocking_agent_name=None)
        for rejected_process_execution in rejected_process_executions:
            if rejected_process_execution in self.agent.process_executions_on_hold:
                del self.agent.process_executions_on_hold[rejected_process_execution]

        # print(get_debug_str(self.agent.name, self.__class__.__name__),
        # "PE on Hold:", len(self.agent.process_executions_on_hold), self.agent.reaction_expected)
        if self.agent.process_executions_on_hold:
            return

        if self.agent.reaction_expected:
            self.reaction_expected = False

            while self.tasks:
                tasks = self.tasks.copy()
                for task in tasks:
                    await task
            self.tasks = []

            await self.agent.change_handler.go_on(agent_name=self.agent.name)

        else:
            self.pre_reaction = True

        # abc = [(pe.identification, pe.process.name, pe.order.identification, name)
        #        for pe, name in self.agent.process_executions_on_hold.items()]
        # print(self.agent.name, "Process Execution on Hold: ", abc)
        # print([(process_execution.identification, process_execution.process.name,
        #         process_execution.order.identification)
        #        for process_execution in self.rejected_process_executions])

    def receive_goods(self, good_receipt_process_executions):
        released_process_executions_sorted = sorted(good_receipt_process_executions,
                                                    key=lambda process_execution: process_execution.executed_start_time)

        for released_process_execution in released_process_executions_sorted:
            self._execute_process(self.agent.name, released_process_execution)

    def inform_actual_process_execution(self, actual_process_execution: ProcessExecution):
        """called by the change_handler to inform about a subscribed process_execution"""

        for resource in actual_process_execution.get_resources():

            if resource in self.agent._resources:
                plan_process_execution_id = actual_process_execution.connected_process_execution.identification
                process_execution_id = actual_process_execution.identification
                resource.update_period_by_actual(start_time=actual_process_execution.executed_start_time,
                                                 end_time=actual_process_execution.executed_end_time,
                                                 process_execution_id=process_execution_id,
                                                 plan_process_execution_id=plan_process_execution_id)

    async def _wait_for_process_execution(self, process_execution):
        iteration = 0

        # print("PE ID awaited: ", self.agent.name, process_execution.process.identification,
        #       process_execution.identification)

        while True:
            self.agent.new_process_executions_available.clear()  # what is meant for it
            # print("new_process_executions_available", process_execution.identification,
            #       process_execution.process.name, self.agent.new_process_executions_available)
            await self.agent.new_process_executions_available.wait()
            self.agent.new_process_executions_available.clear()
            iteration += 1
            if process_execution in self.agent.process_executions_on_hold:
                # print("PE ID available: ", self.agent.name, process_execution.get_name(),
                #       process_execution.identification)

                break

# """
# Build the interface between the negotiation and the real process_execution in the simulation/ shop_floor etc.
# Communication to the env
# @last update: 21.08.2023
# """
#
# # Imports Part 1: Standard Imports
# from __future__ import annotations
#
# import asyncio
# import contextlib
# from typing import TYPE_CHECKING, Optional
#
# # Imports Part 3: Project Imports
# from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour
# from ofact.twin.agent_control.helpers.debug_str import get_debug_str
# from ofact.twin.agent_control.helpers.sort_process_executions import _get_sorted_process_executions
# from ofact.twin.change_handler.reactions import ProcessExecutionAction, Action
#
# # Imports Part 2: PIP Imports
# if TYPE_CHECKING:
#     from datetime import timedelta
#     from ofact.twin.change_handler.change_handler import ChangeHandler
#     from ofact.twin.state_model.processes import ProcessExecution
#     from ofact.twin.state_model.sales import Order
#
#
# class EnvInterfaceBehaviour(DigitalTwinCyclicBehaviour):
#     """
#     The env interface is used as interface between the agent and the environment, respectively the change handler
#     From the change_handler, process_executions are forwarded (if subscribed before). Subsequently, the planning
#     is triggered when changes with relevant effects are available.
#     On the way from the agent to the change handler, process_executions are forwarded or a notice
#     that no action would be taken is passed.
#     """
#
#     def __init__(self):
#         super(EnvInterfaceBehaviour, self).__init__()
#
#     def inform_actual_process_execution(self, actual_process_execution: ProcessExecution):
#         """called by the change_handler to inform about a subscribed process_execution"""
#         # print(self.agent.name, "INFORM")
#         if "order_agent" in self.agent.name:
#             print(self.agent.name, "process finished", actual_process_execution.process.name,
#                   actual_process_execution.connected_process_execution in self.agent.process_executions_in_execution)
#         if actual_process_execution.connected_process_execution in self.agent.process_executions_in_execution:
#             self.agent.process_executions_in_execution.pop(actual_process_execution.connected_process_execution)
#             self.agent.process_executions_finished.append(actual_process_execution.connected_process_execution)
#             # ToDo: intern handling
#         else:
#             pass  # ToDo Case: resource (not main_resource) is subscribed on pe but did not track it in his lists
#
# class OrderPoolEnvInterfaceBehaviour(EnvInterfaceBehaviour):
#
#     def __init__(self):
#         """
#         reaction_expected: a bool that specifies if the change_handler expects an action
#         """
#         super(OrderPoolEnvInterfaceBehaviour, self).__init__()
#         self.reaction_expected: bool = False
#
#         self.current_round = 0  # strongly dependent on the counter of the agents model (find an elegant way)
#
#     def plan_order_release(self, order: Order):
#         self.agent.plan_order_release(order, agent_name=self.agent.name, env_interface_behaviour=self,
#                                       order_subscription= True)
#
#     def reminder_for_order_release(self, order: Order):
#
#         self.agent.reminder_for_order_release(order)
#
#     async def on_end(self):
#         await super().on_end()
#
#
# class OrderEnvInterfaceBehaviour(EnvInterfaceBehaviour):
#
#     def __init__(self):
#         """
#         reaction_expected: a bool that specifies if the change_handler expects an action
#         """
#         super(OrderEnvInterfaceBehaviour, self).__init__()
#         self.reaction_expected: bool = False
#
#         self.current_round = 0  # strongly dependent on the counter of the agents model (find an elegant way)
#
#     async def run(self):
#         await super().run()
#
#         if self.agent.process_executions_queue:
#             self.subscribe_on_ape()
#
#         if self.reaction_expected:
#             # if str(datetime.now().microsecond)[0:3] == "100":
#             #     print(f"{datetime.now()} | [{self.agent.name:35}] Reaction expected")
#             if self.agent.activity != self.agent.Activity.NO_ACTION:
#                 await self.agent.waiting_for_planning_end
#
#             self.reaction_expected = False
#             self.current_round += 1
#             # from datetime import datetime
#             # print(datetime.now(), self.agent.name)
#             await self.agent.change_handler.go_on(agent_name=self.agent.name,
#                                                   round_=self.current_round)
#
#     def subscribe_on_ape(self):
#         process_executions = []
#         while self.agent.process_executions_queue:
#             self_agent_name, planned_process_execution = self.agent.process_executions_queue.pop(0)
#
#             self.agent.change_handler: ChangeHandler
#             self.agent.change_handler.subscribe_on_ape(planned_process_execution=planned_process_execution,
#                                                        agent_name=self_agent_name,
#                                                        env_interface_behaviour=self)
#
#             self.agent.process_executions_in_execution[planned_process_execution] = self_agent_name
#             process_executions.append(planned_process_execution)
#
#         self.agent.process_executions_order.extend(process_executions)
#         self.agent.process_executions_order = _get_sorted_process_executions(self.agent.process_executions_order)
#
#     def inform_actual_process_execution(self, actual_process_execution: ProcessExecution):
#
#         super().inform_actual_process_execution(actual_process_execution=actual_process_execution)
#         self.agent.process_executions_order.remove(actual_process_execution.connected_process_execution)
#         self.agent.process_executions_order.append(actual_process_execution)
#         self.agent.process_executions_order = _get_sorted_process_executions(self.agent.process_executions_order)
#
#     def request_go_on(self):
#         """Called from the change_handler to trigger a reaction to go_on"""
#         self.reaction_expected = True
#         self.agent.waiting_for_planning_end = asyncio.get_event_loop().create_future()
#         self.agent.activity = self.agent.Activity.IN_PLANNING
#         if self.agent.waiting_on_next_round is not None:
#             try:
#                 self.agent.waiting_on_next_round.set_result(True)
#             except asyncio.exceptions.InvalidStateError:
#                 pass
#             self.agent.waiting_on_next_round = None
#         # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Request go on: {self.agent.activity}")
#
#     async def on_end(self):
#         await super().on_end()
#
#
# class ResourceEnvInterfaceBehaviour(EnvInterfaceBehaviour):
#
#     def __init__(self):
#         """
#         process_executions_to_forward: a list of process_executions_components that should be forwarded to another agent
#         """
#         super(ResourceEnvInterfaceBehaviour, self).__init__()
#
#         release_template = {"metadata": {"performative": "inform",
#                                          "ontology": "Release",
#                                          "language": "OWL-S"}}
#         cancel_template = {"metadata": {"performative": "inform",
#                                         "ontology": "Cancel",
#                                         "language": "OWL-S"}}
#         self.templates = [release_template, cancel_template]
#
#         self.metadata_conditions_lst = [{"performative": "inform", "ontology": "Release"},
#                                         {"performative": "inform", "ontology": "Cancel"}]
#
#         self.pre_reaction = False
#
#         self.released_process_executions = []
#         self.task_released_process_executions = None
#         self.rejected_process_executions = []
#         self.rejected_process_executions2 = []
#         self.task_rejected_process_executions = None
#
#         self.tasks = []
#
#         self._release_queue = asyncio.Queue()
#         self._reject_queue = asyncio.Queue()
#
#         self.release_task = None
#         self.reject_task = None
#
#     async def on_start(self):
#         await super().on_start()
#         self.release_task = asyncio.create_task(self._release_loop())
#         self.reject_task = asyncio.create_task(self._reject_loop())
#
#     def initial_subscription(self):
#         """initial subscription on the process_executions_components
#         that include the resources managed by the ResourceAgent"""
#         for resource in self.agent._resources:
#             self.agent.change_handler.subscribe_on_entity(entity=resource,
#                                                           agent_name=self.agent.name,
#                                                           env_interface_behaviour=self)
#
#     async def run(self):
#         await super().run()
#         # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" On hold:",
#         #       [process_execution.identification
#         #        for process_execution, _ in self.agent.process_executions_on_hold.items()])
#
#         msg_process_execution_release_or_rejection_received = \
#             await self.agent.receive_msg(self, timeout=10, metadata_conditions_lst=self.metadata_conditions_lst)
#         # print(get_debug_str(self.agent.name, self.__class__.__name__),
#         #       msg_process_execution_release_or_rejection_received)
#         if msg_process_execution_release_or_rejection_received is not None:
#             msg_content, _, msg_ontology, _ = \
#                 msg_process_execution_release_or_rejection_received
#             if msg_ontology == "Release":
#                 await self._release_queue.put(msg_content)
#             elif msg_ontology == "Cancel":  # Cancel
#                 await self._reject_queue.put(msg_content)
#
#     def request_go_on(self):
#         """Called from the change_handler to trigger a reaction to go_on"""
#         if self.pre_reaction:
#             # print("Pre reaction:", self.agent.name)
#             asyncio.create_task(self.agent.change_handler.go_on(agent_name=self.agent.name))
#             self.pre_reaction = False
#             return
#         else:
#             self.agent.reaction_expected = True
#             # print(f"{datetime.now()} | [{self.agent.name:35}] Request go on")
#
#     async def _release_loop(self):
#         """Verarbeitet nacheinander alle freigegebenen ProcessExecutions."""
#         while True:
#             items = await self._release_queue.get()
#             for pe_dict in items:
#                 await self._process_release(pe_dict)
#
#     async def _reject_loop(self):
#         """Verarbeitet nacheinander alle stornierten ProcessExecutions."""
#         while True:
#             items = await self._reject_queue.get()
#             for pe in items:
#                 await self._process_reject(pe)
#
#     async def _process_release(self, pe_dict):
#         planned_process_execution = pe_dict["ProcessExecution"]
#         if planned_process_execution in self.agent.process_executions_on_hold:
#             await self._wait_for_process_execution(planned_process_execution)
#
#         requester_agent_name = self.agent.process_executions_on_hold.pop(planned_process_execution)
#
#         deviation_tolerance_time_delta = min(self.agent.deviation_tolerance_time_delta,
#                                              pe_dict["Deviation Tolerance Time Delta"])  # ToDo: None problem
#         notification_time_delta = min(self.agent.notification_time_delta,
#                                       pe_dict["Notification Time Delta"])
#
#         action = ProcessExecutionAction(
#             type_=Action.Types.PROCESS_EXECUTION,
#             agent_name=self.agent.name,
#             planned_process_execution=planned_process_execution,
#             env_interface_behaviour=self,
#             ape_subscription=True,
#             deviation_tolerance_time_delta=deviation_tolerance_time_delta,
#             notification_time_delta=notification_time_delta,
#         )
#         self.agent.change_handler.act(action=action)
#         self.agent.process_executions_in_execution[planned_process_execution] = requester_agent_name
#
#     async def _process_reject(self, pe):
#         if pe.main_resource in self.agent.resources and pe not in self.agent.process_executions_on_hold:
#             await self._wait_for_process_execution(pe)
#         # hier Deine Logik zum Zurücksetzen der Planung
#         self.agent.reset_planned_process_executions(
#             unused_process_executions=[pe],
#             blocking_agent_name=None,
#         )
#         self.agent.process_executions_on_hold.pop(pe, None)
#
#     async def _wait_for_process_execution(self, process_execution):
#         iteration = 0
#
#         # print("PE ID awaited: ", self.agent.name, process_execution.process.identification,
#         #       process_execution.identification)
#
#         while True:
#             loop = asyncio.get_event_loop()
#             self.agent.new_process_executions_available = loop.create_future()  # what is meant for it
#             # print("new_process_executions_available", process_execution.identification,
#             #       process_execution.process.name, self.agent.new_process_executions_available)
#             await self.agent.new_process_executions_available
#             self.agent.new_process_executions_available = None
#             iteration += 1
#             if process_execution in self.agent.process_executions_on_hold:
#                 # print("PE ID available: ", self.agent.name, process_execution.get_name(),
#                 #       process_execution.identification)
#
#                 break
#
#     def inform_actual_process_execution(self, actual_process_execution: ProcessExecution):
#         """called by the change_handler to inform about a subscribed process_execution"""
#
#         for resource in actual_process_execution.get_resources():
#
#             if resource in self.agent._resources:
#                 plan_process_execution_id = actual_process_execution.connected_process_execution.identification
#                 process_execution_id = actual_process_execution.identification
#                 resource.update_period_by_actual(start_time=actual_process_execution.executed_start_time,
#                                                  end_time=actual_process_execution.executed_end_time,
#                                                  process_execution_id=process_execution_id,
#                                                  plan_process_execution_id=plan_process_execution_id)
#
#     async def on_end(self):
#         await super().on_end()
#         # 3. Beim Verlassen noch sauberes Herunterfahren
#         self.release_task.cancel()
#         self.reject_task.cancel()
#         with contextlib.suppress(asyncio.CancelledError):
#             await self.release_task
#             await self.reject_task