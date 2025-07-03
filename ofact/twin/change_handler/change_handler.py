"""
Encapsulates the change_handlers that acts as interface between the digital_twin and
the real_world (respectively the simulation) and the agents
The change handler is responsible to ensure that the digital twin is up to date and filled with high quality data.
Tasks:
- digital twin objects are checked on completeness and validity.
- digital twin process_models models are updated

ToDo: [process_models] access point
ToDo: [process_models] What is the data basis and where comes the data basis from ...

ToDo: [orders] @Christian - Where does the orders come from?

ToDo: note from Jonas (Wielage)
 - two DTU can take the same DT-objects - have the same state
 - the two changes the objects in parallel - but sequential is wished
@last update: 21.08.2023
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

# Imports Part 2: PIP Imports
import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, List, Union, Dict, Optional, Literal

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

# Imports Part 3: Project Imports
from ofact.twin.state_model.entities import Part, PassiveMovingResource, Storage, ActiveMovingResource, EntityType
from ofact.twin.state_model.processes import (ProcessTimeController, TransitionController, TransformationController,
                                              QualityController, ResourceController, ProcessController)
from ofact.twin.state_model.sales import Customer, Order
from ofact.env.simulation.event_discrete import EventDiscreteSimulation
# from ofact.twin.change_handler.state_model_consistency.partially_filled_objects_cache import PartiallyFilledObjects
from ofact.twin.change_handler.state_model_consistency.consistency_handler import (BatchConsistencyHandler,
                                                                                   SingleObjectConsistencyHandler)
from ofact.twin.change_handler.reactions import Action, ProcessExecutionAction
from ofact.twin.change_handler.Observer import Observer
if TYPE_CHECKING:
    from datetime import timedelta

    from ofact.twin.agent_control.behaviours.env_release.interface import EnvInterfaceBehaviour
    from ofact.twin.state_model.process_models import DTModel
    from ofact.twin.state_model.entities import Entity
    from ofact.twin.state_model.processes import ProcessExecution
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.agent_control.organization import Agents
    from ofact.env.environment import Environment

    from ofact.twin.change_handler.state_model_consistency.consistency_handler import InconsistencyTypes
logger = logging.getLogger("change_handler")


def _get_re_trainable_process_controllers(digital_twin: StateModel) -> list:
    all_process_controllers = digital_twin.get_all_process_controllers()
    re_trainable_process_controllers = [process_controller
                                        for process_controller in all_process_controllers
                                        if process_controller.model_is_re_trainable()]

    return re_trainable_process_controllers


class ChangeHandler:

    def __init__(self, digital_twin: StateModel, environment: Union[Environment, EventDiscreteSimulation],
                 agents: Agents):
        """
        Responsible for handling of changes, coming from the environment. The changes are checked and
        integrated in the digital twin.

        Parameters
        ----------
        digital_twin: the digital twin model that is changed
        environment: the simulation or data transformation/ integration that have an impact on the digital twin
        and pushes the changes, respectively, the changes can be pulled.
        agents: the agent's model that contains all agents available in the current scenario in use ...
        """
        self._environment: Union[Environment, EventDiscreteSimulation] = environment
        self._digital_twin: StateModel = digital_twin
        self._agents: Agents = agents

        self.observer = Observer()
        self.observer.set_digital_twin(self._digital_twin)

        self._subscriptions_entities: Dict[Entity: List[Literal]] = {}
        self._subscriptions_ape: Dict[ProcessExecution: List[Literal]] = {}
        self._agent_name_interface_behaviours: Dict[Literal: EnvInterfaceBehaviour] = {}

        # self._partially_filled_objects = PartiallyFilledObjects()
        self._batch_consistency_handler = BatchConsistencyHandler(self._digital_twin)
        self._single_object_consistency_handler = SingleObjectConsistencyHandler()

        self._digital_twin_adding_methods_single = \
            {EntityType: self._digital_twin.add_entity_type,
             Customer: self._digital_twin.add_customer,
             Order: self._digital_twin.add_order,
             Part: self._digital_twin.add_part,
             PassiveMovingResource: self._digital_twin.add_passive_moving_resource,
             ActiveMovingResource: self._digital_twin.add_active_moving_resource,
             Storage: self._digital_twin.add_stationary_resource}
        self._digital_twin_adding_methods_batch = \
            {Customer: self._digital_twin.add_customers,
             Order: self._digital_twin.add_orders,
             Part: self._digital_twin.add_parts,
             PassiveMovingResource: self._digital_twin.add_passive_moving_resources,
             ActiveMovingResource: self._digital_twin.add_active_moving_resources,
             Storage: self._digital_twin.add_stationary_resources}
        self._re_trainable_process_controllers: List = _get_re_trainable_process_controllers(digital_twin)


    # # # # process models methods # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def get_process_controllers_with_re_trainable_models(self) -> (
            List[ProcessTimeController], List[TransitionController], List[TransformationController],
            List[QualityController], List[ResourceController]):
        """
        Retrieves the re-trainable process models of various types.

        Returns
        -------
        controllers: A tuple with five lists, each containing the re-trainable process models of a specific type:
            - A list of LeadTimeController instances
            - A list of TransitionController instances
            - A list of TransformationController instances
            - A list of QualityController instances
            - A list of ResourceController instances
        """
        lead_time_controllers = self.get_re_trainable_process_controllers_of_class(ProcessTimeController)
        transition_controllers = self.get_re_trainable_process_controllers_of_class(TransitionController)
        transformation_controllers = self.get_re_trainable_process_controllers_of_class(TransformationController)
        quality_controllers = self.get_re_trainable_process_controllers_of_class(QualityController)
        resource_controllers = self.get_re_trainable_process_controllers_of_class(ResourceController)

        return (lead_time_controllers, transition_controllers, transformation_controllers,
                quality_controllers, resource_controllers)

    def get_re_trainable_process_controllers_of_class(self, process_controller_class: type):
        re_trainable_process_controllers_of_class = [process_controller
                                                     for process_controller in self._re_trainable_process_controllers
                                                     if isinstance(process_controller, process_controller_class)]
        return re_trainable_process_controllers_of_class

    def update_process_models(self, process_controllers: Optional[List[DTModel]] = None,
                              batch_size: Optional[int] = None):
        """
        Update the process models of the digital twin (that can be updated)
        if required (is checked in the retraining_needed request).

        Parameters
        ----------
        process_controllers: controller that control the models ...
        batch_size: The batch size used for retraining the process models.

        Returns
        -------
        successful: True if the process models were updated successfully, False otherwise.
        ToDo: the test_batch should be model specific ...
        """

        process_controllers_to_retrain = []
        for process_controller in process_controllers:
            process_controller: ProcessController
            retraining_needed, retraining_reason = process_controller.model_retraining_needed()  # needed
            print(f"retraining_needed: {retraining_needed}  retraining_reason: {retraining_reason}")

            if retraining_needed:
                process_controllers_to_retrain.append((process_controller, retraining_reason))

        # ToDo: sorting needed?
        process_controllers_sorted = [process_model
                                      for process_model, retraining_reason in process_controllers_to_retrain]

        if tqdm is not None:
            progress_bar = tqdm(process_controllers_sorted)
        else:
            progress_bar = process_controllers_sorted
        for process_controller in progress_bar:
            process_controller.retrain(batch_size=batch_size)
            progress_bar.set_postfix({"DT Process Model Training": "Process Controller {ToDo name}"})

        return True  # return a report ...

    # # # # methods called from agent side # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def planned_events_queue_empty(self):
        return self._environment.planned_events_queue_empty()

    def get_frequency_without_execution(self):
        return self._environment.get_frequency_without_execution()

    def subscribe_on_entity(self, entity: Entity, agent_name: str,
                            env_interface_behaviour: EnvInterfaceBehaviour):
        """Subscribe on the changes of the object"""
        self._subscriptions_entities.setdefault(entity,
                                                []).append(agent_name)
        self._agent_name_interface_behaviours[agent_name] = env_interface_behaviour

    def act(self, action: Action):
        """action from an agent"""
        if action.type_ == Action.Types.PROCESS_EXECUTION:
            action: ProcessExecutionAction
            self.add_planned_process_execution(
                planned_process_execution=action.planned_process_execution,
                agent_name=action.agent_name,
                env_interface_behaviour=action.env_interface_behaviour,
                ape_subscription=action.ape_subscription,
                deviation_tolerance_time_delta=action.deviation_tolerance_time_delta,
                notification_time_delta=action.notification_time_delta
            )
        else:
            raise NotImplementedError

    async def go_on(self, agent_name: str):
        """if the agent did not want to plan after getting actual_process_execution"""
        pass

    def validate_process_chain_consistency(self, process_executions_batch: list[ProcessExecution]):
        return self._batch_consistency_handler.validate_process_chain(process_executions_batch)

    def ensure_open_features_consistency(self, process_executions_batch: list[ProcessExecution]):
        order_not_possible_features = self._batch_consistency_handler.validate_open_features(process_executions_batch)
        for order, not_possible_features in order_not_possible_features.items():
            order: Order
            for feature in not_possible_features:
                order.remove_feature(feature)

    def add_planned_process_execution(self, planned_process_execution: ProcessExecution,
                                      agent_name: Optional[Literal] = None,
                                      env_interface_behaviour: Optional[EnvInterfaceBehaviour] = None,
                                      ape_subscription: bool = False, completely_filled_enforced: bool = True,
                                      deviation_tolerance_time_delta: Optional[timedelta] = None,
                                      notification_time_delta: Optional[timedelta] = None):
        """
        Add a planned_process_execution to the digital_twin and add subscriptions for the agents

        Parameters
        ----------
        planned_process_execution: a process_execution of the event type PLAN
        that will be added to the digital twin.
        agent_name: name of the agent that adds the planned process_execution
        env_interface_behaviour: interface behaviour of the agent used for the communication if subscribed by the agent
        ape_subscription: if True, the agent with the 'agent_name' subscribe on the process_execution
        completely_filled_enforced: if True, an exception is raised if the process execution is not completely filled
        deviation_tolerance_time_delta: if set, the environment is permitted to handle deviations of
        process_execution from the planned executed_start_time within the time_delta
        notification_time_delta: if set, a notification is sent,
        "notification_duration_before_completion" (or less) before the process_execution will be completed.
        """
        logging.debug(f"{datetime.now().time()} | [{'ChangeHandler':35}] Add PLANProcessExecution with ID: "
                      f"'{planned_process_execution.identification}' to digital twin")

        self._single_object_consistency_handler.ensure_process_execution_consistency(
            planned_process_execution, completely_filled_enforced)

        self._digital_twin.add_process_execution(process_execution=planned_process_execution)

        if ape_subscription:
            self.subscribe_on_ape(planned_process_execution, agent_name, env_interface_behaviour)

        self._environment.execute_process_execution(
            process_execution=planned_process_execution, deviation_tolerance_time_delta=deviation_tolerance_time_delta,
            notification_time_delta=notification_time_delta)

    def subscribe_on_ape(self, planned_process_execution, agent_name, env_interface_behaviour):
        """Subscribe on the actual process_execution of the planned one"""
        self._subscriptions_ape.setdefault(planned_process_execution,
                                           []).append(agent_name)
        self._agent_name_interface_behaviours[agent_name] = env_interface_behaviour

    def get_current_time(self):
        """Get the current time from the change_handler"""
        return self._environment.get_current_time()

    def get_current_time64(self):
        return self._environment.get_current_time64()

    def set_current_time(self, new_current_time: datetime):
        self._environment.set_current_time(new_current_time)

    # methods called from the change_handler
    def add_actual_process_execution(self, actual_process_execution: ProcessExecution,
                                     completely_filled_enforced: bool = True):
        """
        Add an actual_process_execution to the digital_twin

        Parameters
        ----------
        actual_process_execution: a process_execution of the event type ACTUAL
        that will be added to the digital twin.
        completely_filled_enforced: if true, an exception is raised if the process execution is not completely filled
        """

        # print("Add Actual Process Execution", actual_process_execution.get_name())

        order = actual_process_execution.order
        logging.debug(f"{datetime.now().time()} | [{'ChangeHandler':35}] Add Actual Process Execution with ID: "
                      f"'{actual_process_execution.identification}'"
                      f"with process name '{actual_process_execution.get_name()}' "
                      f"from order '{order.external_identifications if order else None}' to digital twin")

        self._single_object_consistency_handler.ensure_process_execution_consistency(
            actual_process_execution, completely_filled_enforced)

        self._digital_twin.add_process_execution(process_execution=actual_process_execution)

        self.observer.update_kpi()
        # determine process_execution_subscriber
        planned_process_execution = actual_process_execution.connected_process_execution

        all_subscriber = self.get_subscribers_process_execution(planned_process_execution=planned_process_execution,
                                                                actual_process_execution=actual_process_execution)
        # forward the actual process_execution to the agents subscribed to
        # if "main part transport" in actual_process_execution.get_process_name():
        #     print(all_subscriber, actual_process_execution.order.identification)

        for agent_name in all_subscriber:
            env_interface_behaviour = self._agent_name_interface_behaviours[agent_name]
            # if "order_agent" in agent_name:
            #     print(actual_process_execution.process.name, agent_name, actual_process_execution.order.identification)
            env_interface_behaviour.inform_actual_process_execution(
                actual_process_execution=actual_process_execution)

    def get_subscribers_process_execution(self, planned_process_execution: ProcessExecution,
                                          actual_process_execution: ProcessExecution):
        """:return subscribers"""

        if planned_process_execution in self._subscriptions_ape:
            ape_subscribers = self._subscriptions_ape[planned_process_execution]
            del self._subscriptions_ape[planned_process_execution]
        else:
            ape_subscribers = []

        all_subscriber: List[Literal] = ape_subscribers
        # object_subscriber
        resources = actual_process_execution.get_resources()  # ToDo: maybe other entities relevant
        for resource in resources:
            if resource in self._subscriptions_entities:
                all_subscriber += self._subscriptions_entities[resource]

        all_subscriber = list(set(all_subscriber))

        return all_subscriber

    def add_objects(self, objects: List[Union[Entity, Customer, Order]], object_checking: bool = False):
        """add objects to the digital_twin and inform the subscribed agents if necessary"""
        object_class = objects[0].__class__
        if object_checking:
            for state_model_object in objects:
                self._single_object_consistency_handler.ensure_consistency(state_model_object)

        digital_twin_add_method = self._digital_twin_adding_methods_batch[object_class]
        digital_twin_add_method(objects)

    def add_object(self, object_: Union[Entity, Customer, Order]):
        """add an object to the digital_twin and inform the subscribed agents if necessary"""
        object_class = object_.__class__
        logging.debug(f"{datetime.now().time()} | [{type(self).__class__.__name__:35}] "
                      f"Add {object_class.__name__} with ID: '{object_.identification}' to digital twin")

        self._single_object_consistency_handler.ensure_consistency(object_)
        digital_twin_add_method = self._digital_twin_adding_methods_single[object_class]

        digital_twin_add_method(object_)

    def set_notification_for_executions_ending_soon(self, objects_for_notification: List[ProcessExecution]):
        pass

    def update_object(self, object_: Union[Entity, Customer, Order, ProcessExecution]):
        """Inform about the update of an object (the object is already updated but the information should be passed
        to the relevant parties if necessary) that is already stored in the digital_twin but not completely filled"""
        self._single_object_consistency_handler.ensure_consistency(object_)


class ChangeHandlerPhysicalWorld(ChangeHandler):
    """asynchronous object"""

    def __init__(self, digital_twin: StateModel, environment: Environment, agents: Agents):
        super(ChangeHandlerPhysicalWorld, self).__init__(digital_twin, environment, agents)


class ChangeHandlerSimulation(ChangeHandler):
    """synchronous object"""

    def __init__(self, digital_twin: StateModel, environment: Environment, agents: Agents):
        super(ChangeHandlerSimulation, self).__init__(digital_twin, environment, agents)

        self._agent_responses: dict[str: bool] = {}  # all values must be true to start the simulation again
        self._agent_provider_responses: dict[str: bool] = {}
        self._agent_env_interface_behaviours: dict[str: EnvInterfaceBehaviour] = {}

        self._simulation_results_available = None
        # self.change_occurred_round: int = 0
        # self.change_occurred_await: Dict[int, asyncio.Future] = {0: asyncio.get_running_loop().create_future()}
        # self.change_occurred_await[0].set_result(True)

        self.change_occurred = (None, 0)
        self.actions_batch = []

        self._agent_responses_template = {}
        self._agent_provider_responses_template = {}

        self._checking_responses_active = False
        self._in_simulation = False

    def act(self, action: Action):
        """collect the actions and if all collected they are executed"""
        # print(f"{datetime.now()} | [{'ChangeHandler':35}] Act")

        self.actions_batch.append(action)

    def start_simulation(self):
        """Initial start of the simulation"""
        print(f"{datetime.now().time()} | [{'ChangeHandler':35}] Initialize the simulation start the first time")

        self._request_go_on()

    def add_actual_process_executions(self, actual_process_executions: list[ProcessExecution]):
        """add an actual_process_execution batch to the digital_twin and inform the subscribed agents"""
        for actual_process_execution in actual_process_executions:
            super(ChangeHandlerSimulation, self).add_actual_process_execution(
                actual_process_execution=actual_process_execution)

        self._set_change_occurred()

        # print(f"{datetime.now()} | [{'ChangeHandler':35}] Initialize the simulation start")
        self._request_go_on()

    async def go_on(self, agent_name: str, round_=None):
        """if the agent has no new planned_process_execution after getting actual_process_execution"""
        if self._in_simulation:
            return

        self._agent_responses[agent_name] = True
        # print(f"{datetime.now()} | [{'ChangeHandler':35}] Go on: {agent_name}")
        self._get_change_occurred(round_)

        if self._checking_responses_active:
            return

        all_responses_available = self.check_all_responses_available()

        if not all_responses_available:
            return

        # print(f"{datetime.now()} | [{'ChangeHandler':35}] Simulate")
        if not self._agents.simulation_round_skipped:
            simulation_result = None
            round_ = 0
            self._in_simulation = True
            while simulation_result is None and round_ < 10:
                # print(agent_name, round_, f"Simulate ({self.change_occurred})")
                simulation_result = await self._environment.simulate()

                round_ += 1
                if self._environment.planned_events_queue_empty():
                    break
            self._in_simulation = False
        else:
            self._agents.set_back_simulation_round_skipped()
            simulation_result = None

        if simulation_result is None:  # no actual process_execution given
            self._set_change_occurred()
            self._request_go_on()

    def _get_change_occurred(self, round_):
        if self.change_occurred[0] is None and round_ is not None:
            self.change_occurred = (asyncio.get_running_loop().create_future(), round_)

    def _set_change_occurred(self):
        if self.change_occurred[0] is None:
            return
        if self.change_occurred[0] is not None:
            if not self.change_occurred[0].done():
                self.change_occurred[0].set_result(True)
        self.change_occurred = (None, self.change_occurred[1])

    # def _get_change_occurred(self, round_):
    #     if round_ is not None:
    #         if round_ > self.change_occurred_round:
    #             self.change_occurred_round = round_
    #             self.change_occurred_await[round_] = asyncio.get_running_loop().create_future()
    #
    # def _set_change_occurred(self):
    #     round_ = list(self.change_occurred_await.keys())[-1]
    #     if not self.change_occurred_await[round_].done():
    #         self.change_occurred_await[round_].set_result(True)
    #
    #         if len(self.change_occurred_await) > 5:
    #             del self.change_occurred_await[list(self.change_occurred_await.keys())[0]]
    #
    # async def check_round_finished(self, round_):
    #
    #     change_occurred_round = self.change_occurred_round
    #     round_finished = change_occurred_round < round_
    #     if round_finished:
    #         if not self.change_occurred_await[change_occurred_round].done():
    #             await self.change_occurred_await[change_occurred_round]
    #
    #     return round_finished

    def _request_go_on(self):
        """start the simulation (again)"""
        # request action from agents

        responder_requester_agent_names, responder_provider_agent_names = self._agents.get_responder_agent_names()

        # first the requester agents are asked if they are ready (all requests taken)
        # after it the providers are asked if they are ready (all process_executions submitted)

        agents_env_interface_behaviour, self._agent_responses, self._agent_provider_responses = \
            self._get_request_go_on_information(responder_requester_agent_names, responder_provider_agent_names)

        # trigger response from agents
        for agent_env_interface_behaviour in agents_env_interface_behaviour:
            agent_env_interface_behaviour.request_go_on() # ToDo: prepare - resource digital_twin

    def _get_request_go_on_information(self, responder_requester_agent_names, responder_provider_agent_names):
        agents_env_interface_behaviour = \
            [self._agent_env_interface_behaviours[agent_name] if agent_name in self._agent_env_interface_behaviours
             else self._agents.get_agent_by_name(agent_name).get_env_interface_behaviour()
             # ToDo: prepare - resource digital_twin
             for agent_name in responder_requester_agent_names]  # ToDo: performance

        if len(responder_requester_agent_names) != len(self._agent_responses_template):
            agent_responses = {agent_name: False
                               for agent_name in responder_requester_agent_names}
            self._agent_responses = deepcopy(agent_responses)
        else:
            agent_responses = deepcopy(self._agent_responses)

        if len(responder_requester_agent_names) != len(self._agent_responses_template):
            agent_provider_responses = {agent_name: False
                                        for agent_name in responder_provider_agent_names}
            self._agent_provider_responses = deepcopy(agent_provider_responses)
        else:
            agent_provider_responses = deepcopy(self._agent_provider_responses)

        return agents_env_interface_behaviour, agent_responses, agent_provider_responses

    def check_all_responses_available(self):
        """Check if all agents have responded to the simulation round"""
        self._checking_responses_active = True
        if not (self._agent_responses and all(list(self._agent_responses.values()))):
            self._checking_responses_active = False
            return False

        if not self._agent_provider_responses:
            if self.actions_batch:
                actions_batch = self.actions_batch
                self.actions_batch = []
                for action in actions_batch:
                    super(ChangeHandlerSimulation, self).act(action)
            self._checking_responses_active = False
            return True

        agents_env_interface_behaviour = \
            [self._agent_env_interface_behaviours[agent_name]
             if agent_name in self._agent_env_interface_behaviours
             else self._agents.get_agent_by_name(agent_name).get_env_interface_behaviour()
             for agent_name in list(self._agent_provider_responses.keys())]  # ToDo: prepare - resource digital_twin

        # trigger response from agents
        for agent_env_interface_behaviour in agents_env_interface_behaviour:
            agent_env_interface_behaviour.request_go_on()  # ToDo: prepare - resource digital_twin

        # if len(self._agent_responses.values()) == 1:
        #     planning_of_last_agent = list(self._agent_responses.values())[0]
        #     if planning_of_last_agent:
        #         self._agent_provider_responses = {}
        #         return True

        for agent_name, agent_response in self._agent_provider_responses.items():
            if agent_name not in self._agent_responses:
                self._agent_responses[agent_name] = agent_response

        self._agent_provider_responses = {}

        return self.check_all_responses_available()

    async def check_out(self, agent_name, round_):
        """Order agent has finished his orders"""

        if agent_name in self._agent_responses:
            responder_requester_agent_names, _ = self._agents.get_responder_agent_names()
            while agent_name in self._agent_responses and len(responder_requester_agent_names) > 0:
                # the last agent should not wait
                await self.go_on(agent_name, round_=round_)
                responder_requester_agent_names, _ = self._agents.get_responder_agent_names()
                if agent_name in self._agent_responses:
                    del self._agent_responses[agent_name]

        if agent_name in self._agent_responses:
            del self._agent_responses[agent_name]

        print(f"{datetime.now().time()} | [{'ChangeHandler':35}] {agent_name} checked out")

    async def end_simulation(self, end_simulation_agents=False):
        if not end_simulation_agents:
            return

        await self._agents.end_simulation()
