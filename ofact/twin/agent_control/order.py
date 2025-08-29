"""
This module is for the order_agent and the order_pool_agent.
The oder_agent is responsible for selecting the next order and planning the path
of the product through the production. To achieve this aim the agent communicate with the agents (in particular
resource agents). Another relevant task is the creation of process execution for the monitoring.
@last update: 21.08.2023
"""
# Imports Part 1: Standard Imports
from __future__ import annotations

import asyncio
from copy import copy
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING, Optional

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.twin.agent_control.basic import DigitalTwinAgent
from ofact.twin.agent_control.behaviours.basic import get_templates_tuple
from ofact.twin.agent_control.behaviours.env_release.interface import (OrderEnvInterfaceBehaviour,
                                                                       OrderPoolEnvInterfaceBehaviour)
from ofact.twin.agent_control.behaviours.negotiation.CNET import CNETNegotiationBehaviour
from ofact.twin.agent_control.behaviours.order.management import OrderManagement
from ofact.twin.agent_control.behaviours.order.release import OrderPool
from ofact.twin.agent_control.planning_services.routing import RoutingService
from ofact.twin.agent_control.responsibilities import Responsibility

# ToDo should be an input_param
if TYPE_CHECKING:
    from ofact.twin.change_handler.change_handler import ChangeHandlerSimulation
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.state_model.entities import EntityType, Resource
    from ofact.twin.state_model.processes import ValueAddedProcess, ProcessExecution, Process
    from ofact.twin.state_model.sales import Order
    from ofact.twin.agent_control.organization import Agents

    from ofact.twin.state_model.basic_elements import SourceApplicationTypes


class OrderPoolDigitalTwinAgent(DigitalTwinAgent):
    """
    The order_pool_agent is responsible to respond to new order requests by the order agents.
    This is done through the order release management behaviour, which releases orders after request.
    """

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str, order_pool: list[Order], state_model: StateModel):
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server, ip_address_xmpp_server=ip_address_xmpp_server)
        self.digital_twin = state_model
        self.order_pool = order_pool

        self.OrderPool = OrderPool()

        self.EnvInterface = OrderPoolEnvInterfaceBehaviour()

    def copy(self):
        agent_copy = super(OrderPoolDigitalTwinAgent, self).copy()
        agent_copy.order_pool = agent_copy.order_pool.copy()
        agent_copy.OrderPool = OrderPool()
        agent_copy.EnvInterface = OrderEnvInterfaceBehaviour()

        return agent_copy

    def get_env_interface_behaviour(self):
        """Called from the change_handler to get his communication counter_part behaviour"""
        return self.EnvInterface


    def plan_order_release(self, order: Order):
        self.EnvInterface.plan_order_release(order)

    def reminder_for_order_release(self, order: Order):

        self.OrderPool.reminder_for_order_release(order)

    def set_orders_limit(self, orders_limit):
        self.OrderPool.set_orders_limit(orders_limit)

    def set_order_agent_target_quantity(self, target_quantity):
        self.OrderPool.set_order_agent_target_quantity(target_quantity)

    def set_progress_tracker(self, progress_tracker):
        self.OrderPool.set_progress_tracker(progress_tracker)

    async def setup(self):
        await super().setup()
        # sequence influence not clear

        order_pool_templates = [self.create_template(template)
                                for template in self.OrderPool.templates]

        templates = get_templates_tuple(order_pool_templates)
        self.add_behaviour(self.OrderPool, templates)

        self.add_behaviour(self.EnvInterface)


class OrderDigitalTwinAgent(DigitalTwinAgent):
    """
    The order agent is responsible to complete orders.
    This is mainly done through the order_management behaviour.
    For the communication with the resource agents, who have e.g. the capabilities to assemble parts to the product,
    also a negotiation initiator behaviour is used.
    Last but not least, a process_group_request behaviour is used to request process chains for example for
    a transport process chain.
    For the release of an accepted process batch, an interface behaviour to the simulation is additionally added.
    """

    class Activity(Enum):
        NO_ACTION = "NO_ACTION"
        IN_PLANNING = "IN_PLANNING"

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str,
                 address_book: dict, process_provider: dict, entity_provider: dict[EntityType: list[Resource]],
                 transport_provider: list, processes: dict, value_added_processes: list[ValueAddedProcess],
                 state_model: StateModel, source_application: Optional[SourceApplicationTypes] = None):
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server, ip_address_xmpp_server=ip_address_xmpp_server)
        self.address_book: dict[object, str] = address_book
        self.process_provider = process_provider
        self.entity_provider = entity_provider
        self.transport_provider = transport_provider
        self.processes: dict[str, list[Process]] = processes
        self.value_added_processes: list[ValueAddedProcess] = value_added_processes
        self.digital_twin: StateModel = state_model

        # instantiate routing_service
        resource_types = [self.digital_twin.stationary_resources,
                          self.digital_twin.active_moving_resources,
                          self.digital_twin.passive_moving_resources]
        possible_transport_transfer_processes = self.processes["transport_processes"] + \
                                                self.processes["loading_processes"]
        self.routing_service = RoutingService(resource_types, possible_transport_transfer_processes)

        self.source_application = source_application  # ToDo: instantiation
        self.planning_time_horizont = "planning_time_horizont"  # ToDo: instantiation - period
        self.negotiation_time_limit_period: int = 200  # ToDo: instantiation  # period

        self.feature_process_mapping = self.digital_twin.get_feature_process_mapper()

        self.responsibilities = [Responsibility.MAIN_ENTITY_TRANSPORT]
        # ToDo:
        #   1. Should be an input ...
        #   2. Should be an general attribute for each agent type ...

        # for environment/ simulation
        self.deviation_tolerance_time_delta = timedelta(minutes=5)
        self.notification_time_delta = timedelta(minutes=5)

        # working variables
        self.current_order: None | Order = None
        self.current_work_order = None
        self.current_work_order_preference = None
        self.possible_features = []
        self.chosen_feature = None
        self.next_value_added_process = None
        self.current_process_execution_planned = None
        self.current_process_execution_actual = None

        self.process_executions_queue: list[tuple[str, ProcessExecution]] = []
        self.process_executions_in_execution: dict[ProcessExecution, str | None] = {}
        self.process_executions_finished: list[ProcessExecution] = []
        self.process_executions_order: list[ProcessExecution] = []
        self.waiting_on_next_round = None
        self.waiting_for_planning_end = None
        self.activity = type(self).Activity.IN_PLANNING

        # behaviours
        self.OrderManagement = OrderManagement()

        self.NegotiationBehaviour = CNETNegotiationBehaviour(collection_phase_period=0.1,
                                                             collection_volume_limit=2,
                                                             negotiation_time_limit=self.negotiation_time_limit_period)
        self.change_occurred = (0, None)
        self.EnvInterface = OrderEnvInterfaceBehaviour()

    def set_source_application(self, source_application):
        self.source_application = source_application

    def copy(self):
        agent_copy = super(OrderDigitalTwinAgent, self).copy()
        agent_copy.address_book = agent_copy.address_book.copy()
        agent_copy.process_provider = agent_copy.process_provider.copy()
        agent_copy.entity_provider = agent_copy.entity_provider.copy()
        agent_copy.transport_provider = agent_copy.transport_provider.copy()
        agent_copy.processes = agent_copy.processes.copy()
        agent_copy.value_added_processes = agent_copy.value_added_processes.copy()

        agent_copy.feature_process_mapping = agent_copy.feature_process_mapping.copy()

        # working variables
        agent_copy.possible_features = agent_copy.possible_features.copy()

        agent_copy.process_executions_queue = agent_copy.process_executions_queue.copy()
        agent_copy.process_executions_in_execution = agent_copy.process_executions_in_execution.copy()
        agent_copy.process_executions_finished = agent_copy.process_executions_finished.copy()
        agent_copy.process_executions_order = agent_copy.process_executions_order.copy()
        agent_copy.change_occurred = copy(agent_copy.change_occurred)

        # behaviours
        agent_copy.OrderManagement = OrderManagement()
        agent_copy.NegotiationBehaviour = \
            CNETNegotiationBehaviour(collection_phase_period=0.1,
                                     collection_volume_limit=2,
                                     negotiation_time_limit=self.negotiation_time_limit_period)
        agent_copy.EnvInterface = OrderEnvInterfaceBehaviour()

        return agent_copy

    def get_accepted_time_horizont(self):
        maximal_process_time = max(vap.get_estimated_process_lead_time()
                                   for vap in self.digital_twin.get_value_added_processes())
        maximal_process_time += 350

        return maximal_process_time

    def get_env_interface_behaviour(self):
        """Called from the change_handler to get his communication counter_part behaviour"""
        return self.EnvInterface

    def reserve_time_slots(self, *args):
        pass

    def reserve_parts(self, *args):
        pass

    async def setup(self):
        await super().setup()
        # sequence influence not clear
        order_management_templates = [self.create_template(template)
                                      for template in self.OrderManagement.templates]

        templates = get_templates_tuple(order_management_templates)
        self.add_behaviour(self.OrderManagement, templates)

        negotiation_behaviours_with_templates = self.NegotiationBehaviour.get_behaviour_templates()
        for negotiation_behaviour in negotiation_behaviours_with_templates:
            if "Participant" in negotiation_behaviour["behaviour"].__class__.__name__:
                continue

            templates_lst = [self.create_template(template)
                             for template in negotiation_behaviour["templates"]]

            if not templates_lst:
                self.add_behaviour(negotiation_behaviour["behaviour"])
            else:
                templates = get_templates_tuple(templates_lst)
                self.add_behaviour(negotiation_behaviour["behaviour"], templates)

        self.add_behaviour(self.EnvInterface)

    async def shut_down(self):
        try:
            await asyncio.wait_for(self.agents.check_out(self),timeout=10)
        except:
            print("The task took  more than 10 seconds and is canceled")
        self.change_handler: ChangeHandlerSimulation
        await self.change_handler.check_out(self.name,
                                            round_=self.EnvInterface.current_round + 1)
        simulation_end_conditions_occurred = self.agents.check_simulation_end_conditions_occurred()
        if simulation_end_conditions_occurred:
            print("Not enough orders available")
            await self.agents.end_simulation(self)
            await self.change_handler.end_simulation()
        self.EnvInterface.kill()
        await self.stop()
