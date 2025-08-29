"""

@last update: 21.08.2023
"""
# Imports Part 1: Standard Imports
from __future__ import annotations

import time
import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Tuple, Union

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.twin.agent_control.information_desk import InformationServiceAgent
from ofact.twin.agent_control.order import OrderDigitalTwinAgent, OrderPoolDigitalTwinAgent
from ofact.twin.agent_control.resource import ResourceDigitalTwinAgent
from ofact.twin.agent_control.scheduling_coordinator import SchedulingCoordinatorAgent
from ofact.twin.state_model.helpers.helpers import convert_lst_of_lst_to_lst

if TYPE_CHECKING:
    from ofact.env.environment import Environment
    from ofact.twin.agent_control.helpers.communication_objects import CommunicationObject
    from ofact.twin.agent_control.basic import DigitalTwinAgent

asyncio.PYTHONASYNCIODEBUG = 1  # ToDo: meaning?
logger = logging.getLogger("AgentModel")


class Agents:
    """
    The agents_organization (currently only named as "Agents") is especially used to for the round management.
    The round management is needed because the agents have a naturally asynchronous behaviour.
    To ensure that all agents have the same chance to react and plan, the agents fulfill their tasks
    in synchronous rounds.
    Within the round management, the sync with the order agents as well as with the resource agents itself and
    finally with the coordinator agent is managed.
    Therefore, each agent has access to the organization object.
    Finally, the change_handler use the organization to ensure that all agents have gotten the chance to plan.
    If other tasks are needed to be executed centrally, the organization is as well usable for them.
    For example, for preliminary requests to avoid not promising requests.
    """

    max_round_cycles = 3  # the attribute is used to determine how many cycles/ phases a simulation round includes

    class SimulationEndTypes(Enum):
        """
        The simulation end types are used to determine the termination of the simulation
        :attribute ORDER: a determined amount of orders should be started (normally ended would be better to understand)
        :attribute TIME_LIMIT: a timestamp, when the simulation should end
        """
        ORDER = "ORDER"
        TIME_LIMIT = "TIME_LIMIT"

    def __init__(self, agents: dict = {}, source_agent: dict = {}, sink_agent: dict = {}, progress_tracker=None,
                 simulation_end: tuple[Agents.SimulationEndTypes, str | int] = (SimulationEndTypes.ORDER, 10)):
        """
        The agent model contains the agents and the stored_communication_objects.
        :param agents: a dict that map to each agent type the existing agents (respectively their parameters)
        :param source_agent: can be used to start the simulation
        :param sink_agent: can be used to end the simulation
        :param progress_tracker: responsible to track the progress of the simulation and send it to the frontend
        :param simulation_end:
        - ORDER: last order is completed/ has reached a predefined value (values: int | COMPLETE)
            -> realized in the order_pool agent
        - TIME_LIMIT: time limit has been reached (values: datetime)
        """
        # ensure the right sequence of agent instantiation
        self._agents: dict[DigitalTwinAgent.__class__, list[DigitalTwinAgent]] = agents
        self._agents_lst = [agent for agent_lst in self._agents for agent in agent_lst]
        self._source_agent: dict[DigitalTwinAgent.__class__, list[DigitalTwinAgent]] = source_agent
        self._sink_agent: dict[DigitalTwinAgent.__class__, list[DigitalTwinAgent]] = sink_agent
        self.agents_amount = None
        self.responder_requester_agent_names, self.responder_provider_agent_names = self.get_updated_responder_agents()

        self._agent_name_agent_mapping: dict[str: DigitalTwinAgent] = {}

        self._stored_communication_objects: dict[int, CommunicationObject] = {}

        self.current_round_id = 1
        self._current_round_cycle = 0

        round_ = self.get_current_round()
        self.current_round_id_arisen = {}

        self.agent_callbacks = {}  # for synchronous round_management

        self.skip_complete_round = []

        self.participate_in_end_negotiation = {}
        self.participate_not_in_end_negotiation = {}

        self.scheduling_finished_wait = {}
        self.request_for_coordination = None
        self.scheduling_finished_subscribed = None

        self.possible_proposal_provider = {}
        self.wait_for_agent_planning_proposal_ended = {}
        self.wait_to_next_round = None

        # scenario specific
        self.resources_in_scheduling = []

        # save shut down
        self.xmpp_user_manager = None

        self.simulation_round_skipped = False
        self.skipped_simulation_rounds = 0

        self.number_of_requester_agents_before = None
        self.in_shut_down = False

        self.progress_tracker = progress_tracker

        self.simulation_end: Tuple[Agents.SimulationEndTypes, Union[str, object]] = simulation_end

        # central because of the  transport scheduler
        self.process_executions_projections = {}

        self.order_agent_as_access_point = None

    @property
    def current_round_cycle(self):
        return self._current_round_cycle

    @current_round_cycle.setter
    def current_round_cycle(self, current_round_cycle):
        self._current_round_cycle = current_round_cycle

    def get_current_round(self):
        current_round = (self.current_round_id, self.current_round_cycle)
        return current_round

    def get_current_time(self):
        if self.order_agent_as_access_point is None:
            self.order_agent_as_access_point: OrderDigitalTwinAgent = self.get_agent_of_agent_type("Order")
        try:
            current_time = self.order_agent_as_access_point.change_handler.get_current_time()
        except:
            self.order_agent_as_access_point: OrderDigitalTwinAgent = self.get_agent_of_agent_type("Order")
            current_time = self.order_agent_as_access_point.change_handler.get_current_time()
        return current_time

    def get_current_time64(self):
        if self.order_agent_as_access_point is None:
            self.order_agent_as_access_point: OrderDigitalTwinAgent = self.get_agent_of_agent_type("Order")
        try:
            current_time64 = self.order_agent_as_access_point.change_handler.get_current_time64()
        except:
            self.order_agent_as_access_point: OrderDigitalTwinAgent = self.get_agent_of_agent_type("Order")
            current_time64 = self.order_agent_as_access_point.change_handler.get_current_time64()
        return current_time64

    def add_xmpp_user_manager(self, xmpp_user_manager):
        """Used for the save shut down at the end of the simulation"""
        self.xmpp_user_manager = xmpp_user_manager

    def update_initial(self, environment: Environment):
        self.responder_requester_agent_names, self.responder_provider_agent_names = self.get_updated_responder_agents()
        self._agent_name_agent_mapping = {agent.name: agent for agent_type, agent_lst in self._agents.items()
                                          for agent in agent_lst}

        round_ = self.get_current_round()
        self.agent_callbacks[round_] = self._get_agent_callbacks()

        self._set_simulation_end(environment)

        if self.progress_tracker is not None:
            self.set_progress_tracker(self.progress_tracker)

    def _set_simulation_end(self, environment: Environment):
        type_, param = self.simulation_end

        order_pool_agent: OrderPoolDigitalTwinAgent = self.get_order_pool_agent()
        if type_ == Agents.SimulationEndTypes.ORDER:
            print(f"Order limit: {param}")
            orders_limit, order_target = param
            order_pool_agent.set_orders_limit(orders_limit)

        elif type_ == Agents.SimulationEndTypes.TIME_LIMIT:
            print(f"Time limit: {param}")
            simulation_start, time_limit, order_target = param
            environment.set_time_limit(time_limit)

        else:
            raise Exception

        order_pool_agent.set_order_agent_target_quantity(order_target)

    def set_progress_tracker(self, progress_tracker):
        """Set the progress_tracker"""
        order_pool_agent = self.get_order_pool_agent()

        order_pool_agent.set_progress_tracker(progress_tracker)

    def get_order_pool_agent(self) -> OrderPoolDigitalTwinAgent | None:
        """Return the order_pool_agent if exists or None"""
        return self.get_agent_of_agent_type("OrderPool")

    def get_agent_of_agent_type(self, agent_type_str):
        """Return the agent that has the agent_type_str in the class_name if exists or None"""
        for agent_type, agents in self.agents.items():
            if agent_type_str in agent_type.__name__:
                order_pool_agent = agents[0]
                return order_pool_agent

        return None

    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, new_agents):
        """
        it is important that the agent that starts with the communication is initialized last
        this ensures that the other agents are already built
        in the implementation the order agent start with the initiative and should therefore be initialized last
        :param new_agents: new_agents dict
        """
        independent_agents = {agent_type: agent_lst for agent_type, agent_lst in new_agents.items()
                              if "Order" not in agent_type.__name__}
        dependent_agents = {agent_type: agent_lst for agent_type, agent_lst in new_agents.items()
                            if "Order" in agent_type.__name__}

        # ensure the right sequence of agent instantiation
        self._agents = independent_agents | dependent_agents
        self._agents_lst = [agent for agent_lst in list(self._agents.values()) for agent in agent_lst]
        self._agent_name_agent_mapping = {agent.name: agent for agent_type, agent_lst in self._agents.items()
                                          for agent in agent_lst}

        self.responder_requester_agent_names, self.responder_provider_agent_names = self.get_updated_responder_agents()

        self.agents_amount = len(self.responder_provider_agent_names)

    def get_updated_responder_agents(self):
        responder_requester_agents = \
            convert_lst_of_lst_to_lst([agent_lst for agent_type, agent_lst in self._agents.items()
                                       if issubclass(agent_type, OrderDigitalTwinAgent)])

        responder_provider_agents = \
            convert_lst_of_lst_to_lst([agent_lst for agent_type, agent_lst in self._agents.items()
                                       if issubclass(agent_type, ResourceDigitalTwinAgent)])

        responder_requester_agent_names = [agent.name for agent in responder_requester_agents]
        responder_provider_agent_names = [agent.name for agent in responder_provider_agents]

        return responder_requester_agent_names, responder_provider_agent_names

    def ready(self) -> bool:
        """Determines if the agents are already available"""
        if self.agents:
            return True
        else:
            return False

    def skip_simulation_round(self):
        self.simulation_round_skipped = True
        self.skipped_simulation_rounds += 1

    def set_back_simulation_round_skipped(self):
        self.simulation_round_skipped = False

    def get_agent_by_name(self, agent_name):
        """Get the agent_object by his name"""
        agent = self._agent_name_agent_mapping[agent_name]
        return agent

    def arise_round_id(self):
        """Arise the round id that is responsible for the synchronous progress"""

        if self.current_round_cycle + 1 == type(self).max_round_cycles:
            self.current_round_cycle = 0
            self.current_round_id += 1
            self.skip_complete_round = []

            # if self.current_round_id in self.current_round_id_arisen:
            #     if not self.current_round_id_arisen[self.current_round_id].done():
            #         self.current_round_id_arisen[self.current_round_id].set_result(True)
            #         if len(self.current_round_id_arisen) > 3:
            #             self.current_round_id_arisen.pop(list(self.current_round_id_arisen.keys())[0])

        else:
            self.current_round_cycle += 1

        # print("Current round:", self.get_current_round())

    async def round_arisen(self, round_):
        if round_ + 1 not in self.current_round_id_arisen:
            self.current_round_id_arisen[round_ + 1] = asyncio.get_running_loop().create_future()

        await self.current_round_id_arisen[round_ + 1]

    def arise_round_id_to_round_end(self):
        if self.current_round_cycle == 2:
            return

        if self.current_round_cycle == 0:
            round_ = (self.current_round_id, 1)
            if round_ in self.agent_callbacks:
                self.agent_callbacks[round_] = {key: future_.set_result(None)
                                                for key, future_ in self.agent_callbacks[round_].items()
                                                if not future_.done()}

        elif self.current_round_cycle == 1:
            round_ = (self.current_round_id, 2)
            if round_ in self.agent_callbacks:
                self.agent_callbacks[round_] = {key: future_.set_result(None)
                                                for key, future_ in self.agent_callbacks[round_].items()
                                                if not future_.done()}
        round_ = self.get_current_round()
        new_round = (self.current_round_id, 2)
        self.agent_callbacks[new_round] = self.agent_callbacks[round_]
        if len(self.agent_callbacks.items()) > 6:
            first_key = next(iter(self.agent_callbacks))
            self.agent_callbacks.pop(first_key)
        self.current_round_cycle = type(self).max_round_cycles - 1

    async def ready_for_next_round(self, agent_name, until_end=False):
        """Used to sync the negotiation rounds
        - set the response: all_requests_submitted (the first agent is responsible to start the waiting)"""
        round_ = self.get_current_round()

        if round_ not in self.agent_callbacks:
            self.arise_round_id()
            round_ = self.get_current_round()
            self.agent_callbacks[round_] = self._get_agent_callbacks()
        self.agent_callbacks[round_] = {name: future_
                                        for name, future_ in self.agent_callbacks[round_].items()
                                        if not future_.done()}
        current_agent_callbacks = self.agent_callbacks[round_]
        pending = len(self.agent_callbacks[round_])
        if until_end:
            if agent_name not in self.skip_complete_round:
                self.skip_complete_round.append(agent_name)

        # start a new round
        if not pending:
            self.arise_round_id()

            round_ = self.get_current_round()
            self.agent_callbacks[round_] = self._get_agent_callbacks()

            if self.wait_to_next_round is not None:
                self.wait_to_next_round.set_result(None)
                self.wait_to_next_round = None

            round_ = self.get_current_round()
            current_agent_callbacks = self.agent_callbacks[round_]

        skip_complete_round = False
        # if self already recorded - wait until the round is finished and start a new one
        try:
            current_agent_callbacks[agent_name].done()
        except KeyError:
            raise Exception(current_agent_callbacks, agent_name)
        if current_agent_callbacks[agent_name].done():
            #Jannik:This point is not reached
            await self.wait_on_other_requesters(current_agent_callbacks=current_agent_callbacks)
            round_, skip_complete_round = await self.ready_for_next_round(agent_name)
            return round_, skip_complete_round
        # record self as ready for the next round
        current_agent_callbacks[agent_name].set_result(None)
        # if the round is not finished - wait until the current round is finished and the next round can be started
        self.agent_callbacks[round_] = {name: future_
                                        for name, future_ in self.agent_callbacks[round_].items()
                                        if not future_.done()}
        pending = len(self.agent_callbacks[round_])
        round_pending = pending and \
                        list(current_agent_callbacks.values()) != []

        if round_pending:
            # print(f"not finished: {current_agent_callbacks}")
            await self.wait_on_other_requesters(current_agent_callbacks=current_agent_callbacks)

        if self.skip_complete_round:
            if set(self.skip_complete_round) == set(self.responder_requester_agent_names):
                skip_complete_round = True
                self.arise_round_id_to_round_end()
                round_ = self.get_current_round()

        return round_, skip_complete_round

    def _get_agent_callbacks(self):
        if len(self.agent_callbacks.items()) > 6:
            first_key = next(iter(self.agent_callbacks))
            self.agent_callbacks.pop(first_key)

        agent_callbacks = {agent.name: asyncio.get_running_loop().create_future()
                           for agent in self._agents_lst
                           if not (issubclass(type(agent), SchedulingCoordinatorAgent) or
                                   issubclass(type(agent), InformationServiceAgent) or
                                   issubclass(type(agent), OrderPoolDigitalTwinAgent))}

        return agent_callbacks

    async def wait_on_other_requesters(self, round_=None, current_agent_callbacks=None):
        """Used to sync the negotiation rounds
        - wait until all possible responders have submitted there planning (if wanted)"""
        if current_agent_callbacks is None:
            if round_ is None:
                round_ = self.get_current_round()
            current_agent_callbacks = self.agent_callbacks[round_]

        pending = any([not future_.done() for future_ in list(current_agent_callbacks.values())]) and \
                  list(current_agent_callbacks.values()) != []

        if not pending and self.wait_to_next_round is None:  # first caller
            loop = asyncio.get_running_loop()
            self.wait_to_next_round = loop.create_future()
        if not pending:
            await self.wait_to_next_round
        #print('bevor await loop')
        for keys, callback_to_await in current_agent_callbacks.items():
#            print(f'loop Agent: {keys}')
            await callback_to_await
        #print('after await loop')
        return round_

    def set_await_transport_proposal(self, round_):
        """All first order call_for_proposal requester set an 'await'"""

        if round_ not in self.possible_proposal_provider:
            self.possible_proposal_provider[round_] = {}

    def set_await_proposal(self, possible_proposal_providers, round_):
        """All first order call_for_proposal requester set an 'await'"""

        if round_ not in self.possible_proposal_provider:
            loop = asyncio.get_running_loop()
            self.possible_proposal_provider[round_] = {"callback": loop.create_future()}

        # if len(possible_proposal_providers[0]) == 1:
        #     print(possible_proposal_providers)
        #     import inspect
        #     curframe = inspect.currentframe()
        #     calframe = inspect.getouterframes(curframe, 2)
        #     print(calframe[1][3])
        #     raise Exception

        for possible_proposal_provider in possible_proposal_providers:
            # print(round_, possible_proposal_provider)
            if possible_proposal_provider not in self.possible_proposal_provider[round_]:
                self.possible_proposal_provider[round_][possible_proposal_provider] = 1

                if possible_proposal_provider in self.wait_for_agent_planning_proposal_ended:
                    self.wait_for_agent_planning_proposal_ended[possible_proposal_provider].set_result(True)
                    del self.wait_for_agent_planning_proposal_ended[possible_proposal_provider]

            else:
                self.possible_proposal_provider[round_][possible_proposal_provider] += 1

    async def set_planning_proposal_ended(self, agent_name, round_):
        """All first order proposal responder respond to the 'await' from method set_await_proposal"""
        if agent_name not in self.possible_proposal_provider[round_]:
            loop = asyncio.get_running_loop()
            self.wait_for_agent_planning_proposal_ended[agent_name] = loop.create_future()
            await self.wait_for_agent_planning_proposal_ended[agent_name]

        self.possible_proposal_provider[round_][agent_name] -= 1
        if self.possible_proposal_provider[round_][agent_name] == 0:
            del self.possible_proposal_provider[round_][agent_name]
        # print(self.possible_proposal_provider[round_][agent_name]["number"])

    def set_planning_proposal_ended_eventually(self, round_):
        if len(self.possible_proposal_provider[round_]) == 1:
            if self.possible_proposal_provider[round_]["callback"].done():
                return

            self.possible_proposal_provider[round_]["callback"].set_result(True)

    async def get_planning_proposal_ended_status(self, round_):
        """Return the status - all_planning_proposals processed"""

        if round_ not in self.possible_proposal_provider:
            return True

        await self.possible_proposal_provider[round_]["callback"]

        if self.possible_proposal_provider[round_]:  # reset it if no one else has done it
            self.possible_proposal_provider[round_] = {}
        return True

    def in_end_negotiation(self, round_, agent_name):
        """Inform about the participation in the end round"""
        self.participate_in_end_negotiation.setdefault(round_, set()).update({agent_name})

        if len(self.participate_in_end_negotiation) > 1:
            participants = self.participate_in_end_negotiation[round_]
            self.participate_in_end_negotiation.clear()
            self.participate_in_end_negotiation[round_] = participants
        # print(get_debug_str("AgentModel", "") + f" Round: {round_}, {self.participate_in_end_negotiation[round_]}")

    def not_in_end_negotiation(self, round_, agent_name):
        """Inform about the participation in the end round"""
        self.participate_not_in_end_negotiation.setdefault(round_,
                                                           set()).update({agent_name})

        if len(self.participate_not_in_end_negotiation) <= 1:
            return

        participants = self.participate_not_in_end_negotiation[round_]
        self.participate_not_in_end_negotiation.clear()
        self.participate_not_in_end_negotiation[round_] = participants

    def coordination_requests_complete(self, agent_names: list):
        """Request if all coordination snippets are provided to the coordinator"""

        round_ = list(self.participate_in_end_negotiation.keys())[0]

        if round_ in self.participate_not_in_end_negotiation:
            participate_not_in_end_negotiation = self.participate_not_in_end_negotiation[round_]
        else:
            participate_not_in_end_negotiation = set()

        if len(agent_names) == len(self.participate_in_end_negotiation[round_]) and \
                len(self.participate_in_end_negotiation[round_] | participate_not_in_end_negotiation) == \
                self.agents_amount:
            return True

        else:
            return False

    def get_responder_agent_names(self):
        """Use Case: ChangeHandler want to know which agents should respond if a change occurs"""

        responder_requester_agent_names, responder_provider_agent_names = \
            self.responder_requester_agent_names, self.responder_provider_agent_names
        self.agents_amount = len(responder_provider_agent_names)
        return responder_requester_agent_names, responder_provider_agent_names

    def store_communication_object(self, communication_object: CommunicationObject) \
            -> bool:
        """
        Stores the communication_object in the AgentModel. Therefore, other agents can access to the communication_object,
        if they have the identification_s.
        :param communication_object: one communication_object that should be stored
        :return: True if the communication_object is successfully stored
        ToDo: Should be a ring memory buffer
        """
        if communication_object.identification not in self._stored_communication_objects:
            self._stored_communication_objects[communication_object.identification] = communication_object
        # else:
        # logger.debug(f"{datetime.now()} | [{str('Agents'):35}] "
        #              f"The communication_object with the ID: '{communication_object.identification}' "
        #              f"is already stored in the AgentsModel")

        return True

    def get_communication_object_by_id(self, identification: int) -> CommunicationObject | None:
        """
        Search for the communication_object with the identification (input parameter) and return them.
        :param identification: identification of a communication_objects
        :return: the communication_objects with the identifications (input parameter)
        """

        if identification in self._stored_communication_objects:
            communication_object = self._stored_communication_objects[identification]
        else:
            return None

        return communication_object

    # future music - better way for message processing

    async def send_msg(self, metadata_dict, msg):

        self.msg_send.setdefault(metadata_dict, []).append(msg)

        if metadata_dict in self.wait_on_callbacks:
            self.wait_on_callbacks[metadata_dict].set_result(True)

    async def receive_msg(self, metadata_dict_lst, agent_name):
        """Check if somthing with the metadata is sent"""

        for metadata_dict in metadata_dict_lst:
            self.wait_on.setdefault(metadata_dict, []).append(agent_name)

            if metadata_dict not in self.wait_on_callbacks:
                self.wait_on_callbacks[metadata_dict] = asyncio.get_running_loop().create_future()

        tasks = [asyncio.create_task(self.wait_on_callbacks[metadata_dict])
                 for metadata_dict in metadata_dict_lst]

        done_tasks, pending_tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        msg = self.msg_send[metadata_dict]
        self.wait_on[metadata_dict].remove(agent_name)

    # #### end simulation ####

    def check_out(self, agent_check_out):
        """Check out the order agent, because he has finished his orders"""
        self.responder_requester_agent_names, self.responder_provider_agent_names = self.get_updated_responder_agents()
        self.number_of_requester_agents_before = len(self.responder_requester_agent_names)

        self._agents[agent_check_out.__class__] = \
            [agent
             for agent in self._agents[agent_check_out.__class__]
             if agent != agent_check_out]
        self._agents_lst.remove(agent_check_out)

        self.responder_requester_agent_names, self.responder_provider_agent_names = self.get_updated_responder_agents()
        futures = list(self.agent_callbacks.values())[-2:]
        for future_dict in futures:
            try:
                if agent_check_out.name not in future_dict:
                    continue
                elif future_dict[agent_check_out.name] is None:
                    continue
                elif not future_dict[agent_check_out.name].done():
                    future_dict[agent_check_out.name].set_result(True)
            except:
                pass

        print(f"{datetime.now()} | [{'AgentModel':35}] {agent_check_out.name} checked out")

    def check_simulation_end_conditions_occurred_(self):

        # other rules also 'thinkable'
        responder_requester_agent_names, _ = self.get_responder_agent_names()
        if len(responder_requester_agent_names) < self.number_of_requester_agents_before:  # order agent checked out
            # end of simulation
            return True

        else:
            return False

    def check_simulation_end_conditions_occurred(self):

        # other rules also 'thinkable'
        responder_requester_agent_names, _ = self.get_responder_agent_names()
        if len(responder_requester_agent_names):  # order agent checked out
            # end of simulation
            return False

        else:
            return True

    async def end_simulation(self, agent_shut_down=None):
        """End the simulation means - shut down the agents and save the digital twin in a pickle file"""

        await self._shut_down_agents(agent_shut_down)

    async def _shut_down_agents(self, agent_shut_down=None):
        """Shut down the agents at the end of the simulation run"""
        if self.in_shut_down:
            return

        self.in_shut_down = True
        print("Skipped simulation rounds: ", self.skipped_simulation_rounds)
        print("[organization] stopping Agents...")
        for agent in self._agents_lst:
            if agent == agent_shut_down:
                continue
            try:
                # self.xmpp_user_manager.delete_user(username=agent.name)

                print(f">stopping agent '{agent.name}'...")
                await agent.stop()
            except:
                pass  # already shutted down


# the phase determines how many rounds a simulation round contains
class SinglePhasedAgents(Agents):
    max_round_cycles = 1


class ThreePhasedAgents(Agents):
    max_round_cycles = 3


class StateAgents(Agents):
    """
    State agents are build on a 'current' state of a factory/ shop flor etc. with orders in progress/ not finished.
    These orders are initialized in the first step, done with this extension class.
    """

    def __init__(self, agents: dict = {}, source_agent: dict = {}, sink_agent: dict = {}, progress_tracker=None,
                 simulation_end: tuple[Agents.SimulationEndTypes, str | int] = (Agents.SimulationEndTypes.ORDER, 10)):
        super().__init__(agents=agents, source_agent=source_agent, sink_agent=sink_agent,
                         progress_tracker=progress_tracker, simulation_end=simulation_end)

        self._initial_sync_callbacks = self._get_order_agent_callbacks()

    def _get_order_agent_callbacks(self):
        order_agent_callbacks = {agent.name: asyncio.get_running_loop().create_future()
                                 for agent in self._agents_lst
                                 if issubclass(type(agent), OrderDigitalTwinAgent)}

        return order_agent_callbacks

    async def sync_initial(self, agent_name):
        self._initial_sync_callbacks[agent_name].set_result(True)

        futures = list(self._initial_sync_callbacks.values())
        await asyncio.wait(futures)
        await asyncio.sleep(2)
