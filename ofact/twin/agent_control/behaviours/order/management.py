"""
ToDo: (later) description after implementation ...
handle the order_management
handle the value_added_processes for the order/ production order chosen
Needed access_points to the agent:
next_value_added_process, ... ToDo
@last update: 28.06.2022
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import asyncio
import logging
from copy import copy
from datetime import datetime, timedelta
from math import inf
from random import choices
from typing import TYPE_CHECKING, Dict, Optional, List

# Imports Part 3: Project Imports
import numpy as np
import pandas as pd

from ofact.twin.settings import FILE_LOG_LEVEL, CONSOLE_LOG_LEVEL
from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour
from ofact.twin.agent_control.behaviours.negotiation.objects import CallForProposal, \
    ProcessCallForProposal, ProcessGroupCallForProposal
from ofact.twin.agent_control.behaviours.planning.tree.preference import EntityPreference
from ofact.twin.agent_control.behaviours.planning.tree.preference import arr_sequence_detection
from ofact.twin.agent_control.helpers.communication_objects import ListCO, ObjectCO, \
    AvailabilityInformationCO
from ofact.twin.agent_control.helpers.debug_str import get_debug_str
from ofact.twin.agent_control.helpers.sort_process_executions import _get_sorted_process_executions
from ofact.twin.agent_control.responsibilities import Responsibility
from ofact.twin.state_model.entities import StationaryResource, NonStationaryResource, Storage, ConveyorBelt, \
    ActiveMovingResource
from ofact.twin.state_model.processes import (WorkOrder, ProcessExecution, EntityTransformationNode, Process,
                                              ValueAddedProcess)

# Imports Part 2: PIP Imports
if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.state_model.entities import EntityType, Entity
    from ofact.twin.state_model.sales import Order, Feature

from ofact.twin.utils import setup_dual_logger


# Module-Specific Constants
# logger = logging.getLogger("OrderManagement")


# ToDo: Replace the last_process_executions_tuple - Why is a tuple needed?


def append_provider_to_process_executions(process_executions, provider):
    return [(process_execution, provider)
            for process_execution in process_executions]


def convert_to_lst_with_first_tup_elem(list_with_tuples):
    """Convert a list of tuples to a list with only the first element of each tuple"""
    return [tuple_[0] for tuple_ in list_with_tuples]


def _get_next_origin(process_execution):
    if isinstance(process_execution.destination, StationaryResource):
        next_origin = process_execution.destination
    else:
        next_origin = process_execution.origin
    return next_origin


def _determine_best_proposal(proposals):
    best_time_slot_value_proposal = None
    best_end_time = None
    best_time_slot_value = 0
    best_time_slot_cost_min = float("inf")
    best_duration = None
    # own preferences - more elegance needed
    # preference_values = proposals[0][0].call_for_proposal.preference.preference_values
    best_price = +inf
    for proposal, provider in proposals:
        price = proposal.get_price_over_time()

        if price < best_price:
            best_price = price
            best_proposal = proposal
            best_duration = best_proposal.reference_preference.expected_process_execution_time
            best_time_slot_value = 1
            best_time_slot_value_proposal = [proposal, provider]
        # proposal_preference = \
        #     preference_values.loc[np.intersect1d(prices_combined.index.to_numpy(),
        #                                          preference_values.index.to_numpy(),
        #                                          assume_unique=True)]
        #
        # sequence_indexes = arr_sequence_detection((proposal_preference.index.to_numpy(dtype="datetime64") -
        #                                            np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
        #
        # # + 1 (if a process_execution takes 1 second it means - from second 1 to 2 and not 1 to 1)
        # rolled_dfs = \
        #     [proposal_preference.iloc[int(start_period):
        #                               int(end_period + 1), :].rolling(window=duration, min_periods=duration).sum()
        #      for start_period, end_period in sequence_indexes]
        #
        # for rolled_df in rolled_dfs:
        #     # take the time_slot that is most preferred and "consume" the minimum costs
        #     if best_time_slot_value <= rolled_df.max()[0] and \
        #             prices_combined[rolled_df.idxmax()][0] < best_time_slot_cost_min:
        #         # price is always to the next second + 1 second
        #         if duration > 0:
        #             best_end_time = rolled_df.idxmax().to_numpy()[0] + np.timedelta64(1, "s")
        #         else:
        #             best_end_time = rolled_df.idxmax().to_numpy()[0]
        #         best_time_slot_value = rolled_df.max()[0]
        #         best_time_slot_cost_min = prices_combined[rolled_df.idxmax()][0]
        #         best_time_slot_value_proposal = (proposal, provider)
        #         best_time_slot_value_rolled_dfs = rolled_dfs
        #         best_duration = duration

    # if best_end_time is not None:
    #     best_start_time = best_end_time - np.timedelta64(int(best_duration), "s")

    # still_accepted_time_periods = \
    #     [period
    #      for rolled_df in best_time_slot_value_rolled_dfs
    #      for period in self.get_still_accepted_time_periods(
    #         rolled_df.loc[rolled_df["preference_values"] >= best_time_slot_value * 0.95].index.to_numpy(),
    #         best_duration)]
    # else:
    #     best_start_time = None
    still_accepted_time_periods = []

    best_time_slot_cost_min = best_price
    best_time_slot = best_proposal.reference_preference.get_accepted_time_periods()[0]

    return best_time_slot_value_proposal, best_time_slot, best_time_slot_value, best_duration, \
        best_time_slot_cost_min, still_accepted_time_periods


def _differentiate_proposals(proposals, best_time_slot_value_proposal, best_time_slot, best_time_slot_value,
                             best_duration, best_time_slot_cost_min, still_accepted_time_periods):
    # prepare the best proposal for further negotiation
    if best_time_slot_value > 0 or best_duration == 0:
        best_time_slot_value_proposal[0].add_preferred_time_slot(best_time_slot)
        best_time_slot_value_proposal[0].add_price_preferred(best_time_slot_cost_min)

        # all other proposal are refused
        proposals_to_refuse = [(proposal, provider) for proposal, provider in proposals
                               if proposal.identification != best_time_slot_value_proposal[0].identification]
    else:
        proposals_to_refuse = proposals

    if best_time_slot_value_proposal is None:
        best_time_slot_value_proposal = []
    else:
        best_time_slot_value_proposal = [best_time_slot_value_proposal]

    return best_time_slot_value_proposal, proposals_to_refuse


def get_still_accepted_time_periods(times, duration):
    periods = arr_sequence_detection((times - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
    still_accepted_periods = [(times[int(start_period)] - np.timedelta64(duration, 's'), times[int(end_period)])
                              for start_period, end_period in periods]
    return still_accepted_periods


def _choose_best_proposals(proposals, requests, agent_name):
    """Choose the best proposal setting in accepted time periods
    - for each request a proposal is chosen"""

    if not proposals:
        return [], []

    requests_proposals, requests = map_request_proposal(proposals, requests, agent_name=agent_name)

    # if requests:
    #     print  # raise NotImplementedError("One request is not fulfilled - how to handle?")

    best_time_slot_value_proposal = []
    proposals_to_refuse = []
    for request, request_proposals in requests_proposals.items():
        best_time_slot_value_proposal_batch, best_time_slot, best_time_slot_value, best_duration, \
            best_time_slot_cost_min, still_accepted_time_periods = _determine_best_proposal(request_proposals)

        best_time_slot_value_proposal_batch, proposals_to_refuse_batch = \
            _differentiate_proposals(request_proposals, best_time_slot_value_proposal_batch, best_time_slot,
                                     best_time_slot_value, best_duration, best_time_slot_cost_min,
                                     still_accepted_time_periods)
        if best_time_slot_value_proposal_batch:
            if best_time_slot_value_proposal_batch[0] not in best_time_slot_value_proposal:
                best_time_slot_value_proposal.extend(best_time_slot_value_proposal_batch)
        proposals_to_refuse.extend(proposals_to_refuse_batch)

    proposals_to_refuse = list(set(proposals_to_refuse))

    return best_time_slot_value_proposal, proposals_to_refuse


def map_request_proposal(proposals, requests, agent_name):
    request_proposals = {}

    for proposal, provider in proposals:

        request_processes = [process_execution.process
                             for process_execution in proposal.call_for_proposal.get_process_executions()]
        for request in request_processes:
            if request in requests:
                requests.remove(request)
            elif request not in request_proposals:
                raise Exception([r.name for r in requests], request.name, request_proposals,
                                proposal.call_for_proposal.sender_name, agent_name)
            request_proposals.setdefault(request,
                                         []).append((proposal, provider))

    return request_proposals, requests


class OrderManagement(DigitalTwinCyclicBehaviour):
    """
    Complete the order in negotiation rounds.
    The order management is used to complete order an order.
    But before an order can be completed, the order must be requested from the order pool agent.
    This is therefore the first step.
    Afterward, the next to process value added processes (it could be more than one process -
    but maybe the further implementations works
    not in all cases with more than one process) are determined based on the priority chart.
    The determination rule should be set for each
    project. Maybe a standard pool should be implemented in the future.
    If the value_added_processes  are chosen, preconditions are derived. Preconditions are the main (part/ resource)
    entity_type and the support entity_type which should be organized in advance in the organization.
    The organization is also project dependent. For the currently implemented simulations two different approaches are
    used. For the bicycle world, a simulation round contains three phases
    (each is a round for the negotiation behaviours).
    In the first phase, the main entity_type and support entity_type is organized.
    If it is organized, the phase is skipped for the further simulation rounds of the simulation.
    In the second phase, the value added processes are organized. Here, the max transport time is considered.
    In the future, the time could be destination dependent.
    If the value added process is successfully organized too, the transport is requested (third phase).
    If all phases are successful, the process_executions are released, else the process_executions are thrown away.
    In both cases, the participating agents are informed.
    Currently the organization phases are repeated until a solution is found, without stopping for execution.
    To ensure the feasibility, the consideration period is extended.
    For the Schmaus case, only one phase is needed. This is the case, because the shop floor has
    a low degree of flexibility.
    The main loading needs no access through a resource, to transport the box, etc.,
    the next value added process is fixed and
    no flexibility in the next station possible to approachable/ accessible is available.
    The sequence is a little bit different. First, the support and main entity_type is organized, than the transport,
    and only in the end, the value added process is organized.
    """

    MAX_PHASES = 3

    def __init__(self):
        super(OrderManagement, self).__init__()

        self.value_added_processes_preconditions: dict[ValueAddedProcess: dict] = {}
        self.value_added_process_process_executions: dict[ValueAddedProcess: list[ProcessExecution]] = {}
        self.value_added_process_executions_finished: dict[ValueAddedProcess: list[ProcessExecution]] = {}
        self.value_added_processes_organized_preconditions: dict[ValueAddedProcess: dict] = {}
        self.current_planned_value_added_process: ValueAddedProcess | None = None
        self.planned_process_executions: dict[ValueAddedProcess: list[ProcessExecution]] = {}
        self._requested_times_before = {}
        self._old_value_added_process_execution = None
        self.current_location = None
        self.current_resource_binding = None
        self.reset_current_resource_binding = False
        self.current_time64 = None
        self.current_round = 0
        self.phase = 0

        self.negotiation_failed = 0
        self.amount_negotiation = 0
        self.last_pe = 0
        self.time = pd.Timedelta(seconds=0)

        self.next_request = None
        self.availability_request_level = 1

        self._main_organization_needed = {}
        self.value_added_process_call_for_proposals = None

        self.support_resource = None
        self.main_entity = None  # ToDo

        self.logging = setup_dual_logger()

        # ToDo: Thought support - work order simulation wie bei dynamic attributes

        proposal_inform_template = {"metadata": {"performative": "inform",
                                                 "ontology": "PROPOSAL",
                                                 "language": "OWL-S"}}
        new_order_inform_template = {"metadata": {"performative": "inform",
                                                  "ontology": "ORDER",
                                                  "language": "OWL-S"}}
        availability_inform_result_template = {"metadata": {"performative": "inform-result",
                                                            "ontology": "AVAILABILITY",
                                                            "language": "OWL-S"}}
        self.templates = [proposal_inform_template, new_order_inform_template, availability_inform_result_template]

    async def on_start(self):
        await super().on_start()
        self.current_time64 = self.agent.change_handler.get_current_time64()

    async def run(self):
        await super().run()
        # idea: organize the complete value_added_process in one run
        # print("OrderManagement running successful")
        continue_ = await self.wait_until_start()

        if not continue_:
            return
        # print("Order Round")
        if self.agent.current_order is None:  # query if an order exists
            order_request_time = await self._organize_order()
            if order_request_time is not None:
                self.current_time64 = np.datetime64(order_request_time)
                await self._skip_negotiation_round(amount=type(self).MAX_PHASES, until_end=True)

                self.agent.waiting_for_planning_end.set_result(True)
                self.agent.waiting_on_next_round = asyncio.get_event_loop().create_future()
                self.agent.activity = self.agent.Activity.NO_ACTION
                return

        # await self.agent.change_handler.get_simulation_results_available()
        new_value_added_process, value_added_processes, old_value_added_process_execution = \
            self._value_added_process_selection()

        if not value_added_processes and self.agent.current_order:
            await self._close_order()
            order_request_time = await self._organize_order()
            if order_request_time is not None:
                self.current_time64 = np.datetime64(order_request_time)
                await self._skip_negotiation_round(amount=type(self).MAX_PHASES, until_end=True)

                self.agent.waiting_for_planning_end.set_result(True)
                self.agent.waiting_on_next_round = asyncio.get_event_loop().create_future()
                self.agent.activity = self.agent.Activity.NO_ACTION
                return

            new_value_added_process, value_added_processes, old_value_added_process_execution = (
                self._value_added_process_selection())

        # no planning needed
        if not new_value_added_process or not value_added_processes:
            # ToDo: skip three rounds - acceleration
            await self._skip_negotiation_round(amount=type(self).MAX_PHASES, until_end=True)

            self.agent.waiting_for_planning_end.set_result(True)
            self.agent.waiting_on_next_round = asyncio.get_event_loop().create_future()
            self.agent.activity = self.agent.Activity.NO_ACTION
            return

        self.agent.activity = self.agent.Activity.IN_PLANNING
        # ToDo: Use information from the last value_added_process
        # derive preconditions for the value_added_process
        self.derive_preconditions_value_added_process(value_added_processes=value_added_processes)

        current_time64 = self.agent.change_handler.get_current_time64()
        if current_time64 > self.current_time64:
            self.current_time64 = current_time64

        successful, requested_times_before = \
            await self._organize(value_added_processes=value_added_processes,
                                 old_value_added_process_execution=old_value_added_process_execution)

        if not successful:
            self._requested_times_before = requested_times_before  # used for successor requests
            self._old_value_added_process_execution = old_value_added_process_execution

            # planning horizont before is completely planned and no new process_execution is planned anymore ...
            frequency_without_execution = self.agent.change_handler.get_frequency_without_execution()
            if frequency_without_execution > (10 * self.availability_request_level):
                # if self.agent.current_order:
                #     print("Features requested:", [feature.name
                #                                   for feature in self.agent.current_order.features_requested])
                # used to handle time periods such as breaks or time nobody is working (weekend, night)
                # these times are stepped over

                earliest_possible_execution_times = await self._request_availability(value_added_processes)
                # print("Earliest_possible_execution_times", earliest_possible_execution_times)
                if earliest_possible_execution_times is not None:
                    next_possible_execution_start_time = \
                        self._determine_next_possible_execution_start_time(earliest_possible_execution_times)
                    # update the current time of the order, meaning that the request are now on a new time
                    print("Earliest_possible_time", next_possible_execution_start_time)
                    self.value_added_process_call_for_proposals = None
                    self.current_time64 = np.datetime64(next_possible_execution_start_time)

                    self.availability_request_level += 0.5
        else:
            self.availability_request_level = 1

        # if value_added_processes[0] in self.value_added_process_process_executions:
        #     for pe, _ in self.value_added_process_process_executions[value_added_processes[0]]:
        #         print(pe.get_process_name(), [r.identification for r in pe.get_resources()],
        #               [p.identification for p in pe.get_parts()],
        #               pe.executed_start_time, pe.executed_end_time, pe.order.identification)
        # the value_added_process_eval was successful
        # logger.debug(get_debug_str(self.agent.name, self.__class__.__name__) +
        #              f"{len(self.value_added_process_process_executions[value_added_process])} PE's")

        self.agent.waiting_for_planning_end.set_result(True)
        self.agent.waiting_on_next_round = asyncio.get_event_loop().create_future()
        self.agent.activity = self.agent.Activity.NO_ACTION

    async def wait_until_start(self):

        # while True:
        #     round_ = copy(self.agent.agents.current_round_id)
        #     round_finished = await self.agent.change_handler.check_round_finished(round_)
        #     if round_finished:
        #         self.current_round = round_
        #         break
        #
        #     await self.agent.agents.round_arisen(round_)
        #     # ensure that the agent is not two times in the same round - await round changed
        #
        # print("DEBUBD", self.agent.name, self.agent.agents.current_round_id)

        self.current_round = self.agent.agents.current_round_id

        if self.agent.waiting_on_next_round is not None:
            await self.agent.waiting_on_next_round

        if self.agent.change_handler.change_occurred[1] < self.current_round:
            if self.agent.change_handler.change_occurred[0] is not None:
                if not self.agent.change_handler.change_occurred[0].done():
                    await self.agent.change_handler.change_occurred[0]
                    await self.run()  # ensure the right priority
                    return False

            return True
        else:
            return False  # ensure that the agent is not two times in the same round - await round changed

    def ultimate_fail(self):
        # Means that all order agents fail or only self.agent (scenario dependent)
        return True

    # #### ORDER MANAGEMENT ############################################################################################

    async def _organize_order(self):
        self.logging.info(f'{self.agent.name} Organizing order', extra={"obj_id": self.agent.name})
        # set the next order
        await self._request_new_order()
        print(f'{self.agent.name} Order request finished')
        order_or_datetime = await self._receive_new_order()
        print(f'{self.agent.name} Order datetime {order_or_datetime}')
        if isinstance(order_or_datetime, datetime):
            return order_or_datetime

        self.agent.current_order = order_or_datetime

        order = self.agent.current_order
        if order is None:
            return None

        order_str = str(order.identification)
        if order.external_identifications:
            external_order_names = str([order_name
                                        for external_order_names in list(order.external_identifications.values())
                                        for order_name in external_order_names])
            order_str += " - " + external_order_names

        print(get_debug_str(self.agent.name, self.__class__.__name__) +
              f" The order {order_str} was chosen to execute!")
        self.logging.info(f'{self.agent.name} has chosen Order {order.identification} to execute!',
                          extra={"obj_id": self.agent.name})
        # creation of the production order
        self.agent.current_work_order, self.agent.current_work_order_preference = \
            self.get_next_work_order(sales_order=self.agent.current_order)

        if self.agent.current_order.products:
            await self._handle_orders_already_started_before()
        else:
            print("Order has no product:", str(self.agent.current_order))

        if self.current_round != 0:
            return None

        if hasattr(self.agent.agents, "sync_initial"):
            self.agent.agents.sync_initial(self.agent.name)

        return None

    async def _request_new_order(self):
        """Request the new order from the order pool agent"""

        if "order_pool_agent" not in self.agent.address_book:
            raise Exception

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Request new order")
        providers = [self.agent.address_book["order_pool_agent"]]

        await self.agent.send_msg(behaviour=self, receiver_list=providers, msg_body="",
                                  message_metadata={"performative": "request",
                                                    "ontology": "ORDER",
                                                    "language": "OWL-S"})

    async def _receive_new_order(self) -> Order | None | datetime:
        """Receive the new order from the order pool agent"""

        msg_received = await self.agent.receive_msg(self, timeout=10,
                                                    metadata_conditions={"performative": "inform",
                                                                         "ontology": "ORDER",
                                                                         "language": "OWL-S"})

        if msg_received is not None:
            msg_content, msg_sender, msg_ontology, msg_performative = msg_received

            if msg_content is None:
                print("Kill the order agent:", msg_content, msg_sender, msg_ontology, msg_performative)
                self.kill()
                msg_content = None

            order_or_datetime = msg_content
        else:
            order_or_datetime = await self._receive_new_order()

        return order_or_datetime

    def get_next_work_order(self, sales_order) -> (WorkOrder, EntityPreference):
        """
        Creates a production order based on the sales order and preference too.
        :param sales_order: the feature based order (nearer to the customer view)
        :return work_order: a production order
        :return work_order_preference: an object that manages the preferences for the work_order
        """
        # features without value_added_process(es) are marked as done before the work_order is created
        features_with_value_added_processes = sales_order.get_features_with_value_added_processes()

        value_added_processes_requested = \
            WorkOrder.convert_features_to_value_added_processes_requested(features_with_value_added_processes,
                                                                          self.agent.feature_process_mapping)

        # needed for more than one process associated with the same feature
        value_added_processes_completed: dict[Feature: dict[int: list[ValueAddedProcess]]] = {}
        for feature, process_executions in sales_order.feature_process_execution_match.items():
            process_process_executions = {}
            for process_execution in process_executions:
                process_process_executions.setdefault(process_execution.process,
                                                      []).append(process_execution)
            if not process_process_executions:
                continue

            value_added_processes_completed[feature] = {}
            for process, process_executions_p in process_process_executions.items():
                for index, process_execution in enumerate(process_executions_p):
                    value_added_processes_completed[feature].setdefault(index,
                                                                        []).append(process_execution.process)
                    if feature in value_added_processes_requested:
                        if index in value_added_processes_requested[feature]:
                            value_added_processes_requested[feature][index].remove(process_execution.process)

        # create the production order
        work_order = WorkOrder(value_added_processes_completed=value_added_processes_completed,
                               value_added_processes_requested=value_added_processes_requested,
                               order=sales_order)

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" work_order ready")

        # create a preference_object for the work_order
        accepted_time_horizont = self.agent.get_accepted_time_horizont()  # ToDo: set a rational value
        work_order_preference = EntityPreference([work_order], accepted_time_horizont)
        return work_order, work_order_preference

    def _set_order_product(self, process_execution):
        """Set the product of the order"""

        main_entity = self.check_product_ability(process_execution)
        if main_entity is not None:
            if main_entity not in self.agent.current_order.products:
                self.agent.current_order.products.append(main_entity)

                if len(self.agent.current_order.products) > 1:
                    print(f"Warning: The order products length is maybe too long: "
                          f"{len(self.agent.current_order.products)}")

    async def _close_order(self):
        """
        The method is used to close the order which means that the appropriate variables are set back (and
        the order is written in the archive?)
        :return
        """
        print(get_debug_str(self.agent.name, self.__class__.__name__) +
              f" The order (Order-ID: {self.agent.current_order.identification}) was finished!")
        amount_products_required = len(self.agent.current_order.product_classes)
        self.agent.current_order.products = self.get_products(amount_products_required)

        current_order: Order = self.agent.current_order
        current_order.complete(delivery_date_actual=self.get_delivery_date_actual())

        if self.current_resource_binding is not None:
            await self._reset_resource_binding()
        self.agent.current_order = None
        self.agent.current_work_order = None
        self.agent.process_executions_finished = []
        self.value_added_process_process_executions = {}
        self.value_added_process_executions_finished = {}
        self.value_added_processes_preconditions = {}
        self.planned_process_executions = {}
        self._requested_times_before = {}
        self.reset_current_resource_binding = False
        self.current_location = None

        self.current_planned_value_added_process = None
        self._old_value_added_process_execution = None

        self.support_resource = None
        self.main_entity = None
        self.agent.process_executions_order = []

        return True

    def get_products(self, amount_products_required: int) -> list[Entity]:

        value_added_process_executions = \
            [process_execution
             for process, provider_process_execution_lst in self.value_added_process_executions_finished.items()
             for process_execution, _ in provider_process_execution_lst
             if process_execution.process == process]

        products = []
        for value_added_process_execution in value_added_process_executions:
            main_entity = self.check_product_ability(value_added_process_execution)
            if (main_entity is not None and
                    main_entity not in products):
                products.append(main_entity)

                if len(products) >= amount_products_required:
                    return products

        return products

    def get_delivery_date_actual(self):
        current_time = self.agent.change_handler.get_current_time()

        return current_time

    def check_product_ability(self, process_execution) -> None | Entity:
        main_entity: Entity = process_execution.get_main_entity()
        product_entity_types_needed: [EntityType] = self.agent.current_order.product_classes
        for product_entity_type_needed in product_entity_types_needed:
            if main_entity is not None:
                if product_entity_type_needed.check_entity_type_match_lower(main_entity.entity_type):
                    return main_entity

        return None

    async def _reset_resource_binding(self):

        resource = self.current_resource_binding[0]
        provider = self.agent.address_book[resource]
        resource_binding_object = ObjectCO((self.current_resource_binding[0], None))
        self.agent.agents.store_communication_object(resource_binding_object)
        msg_content = resource_binding_object.identification
        msg_resource_binding_sent = self.agent.create_message({"to": provider,
                                                               "body": msg_content,
                                                               "metadata": {"performative": "inform",
                                                                            "ontology": "ResourceBinding",
                                                                            "language": "OWL-S"}})
        await self.send(msg_resource_binding_sent)

    # #### VALUE_ADDED_PROCESS MANAGEMENT ##############################################################################

    def _value_added_process_selection(self):
        """Select the next value_added_process if possible"""

        value_added_process = self.current_planned_value_added_process
        if value_added_process not in self.value_added_process_process_executions:
            value_added_process = None

        if value_added_process is not None:
            new_value_added_process = self.check_value_added_process_finished()

            if new_value_added_process:
                self._old_value_added_process_execution = self.get_value_added_process_execution()
                self.finish_value_added_process()
                # print(get_debug_str(self.agent.name, self.__class__.__name__) +
                #       f" Value Added Process {self._old_value_added_process_execution.process.name} completed")

        else:
            new_value_added_process = True

        if not (self.agent.current_work_order and new_value_added_process):
            if value_added_process:
                value_added_processes = [value_added_process]
            else:
                value_added_processes = []
            return new_value_added_process, value_added_processes, self._old_value_added_process_execution

        # Choose the next current value added process  # ToDo: (later) maybe parallel
        value_added_processes = self.get_next_value_added_processes()
        if value_added_processes is not None:
            self.logging.info(
                f'Order ID:{self.agent.current_order.identification} Value added process selection: {value_added_processes[0].name}',
                extra={"obj_id": self.agent.name})
        else:
            self.logging.info(
                f'Order ID:{self.agent.current_order.identification} Value added process is {value_added_processes}',
                extra={"obj_id": self.agent.name})
        return new_value_added_process, value_added_processes, self._old_value_added_process_execution

    def check_value_added_process_finished(self) -> bool:
        """Check if all process_executions_components for the value_added_process are available"""

        if self.current_planned_value_added_process not in self.value_added_process_process_executions:
            return False

        value_added_process_executions = \
            self.value_added_process_process_executions[self.current_planned_value_added_process]
        planned_process_executions_with_provider = \
            [planned_process_execution
             for planned_process_execution, provider in value_added_process_executions]

        planned_process_executions_finished = self.agent.process_executions_finished

        finished_process_executions = \
            set(planned_process_executions_with_provider).intersection(
                set(planned_process_executions_finished))

        if len(finished_process_executions) == len(planned_process_executions_with_provider):

            return True
        else:
            return False

    def finish_value_added_process(self):
        """
        Complete a value_added_process in the work/sales_order
        """
        value_added_process_executions = \
            self.value_added_process_process_executions[self.current_planned_value_added_process]
        self.value_added_process_executions_finished.setdefault(self.current_planned_value_added_process,
                                                                []).extend(value_added_process_executions)
        del self.value_added_process_process_executions[self.current_planned_value_added_process]
        process_executions = convert_to_lst_with_first_tup_elem(value_added_process_executions)
        self.agent.current_work_order.complete_value_added_process(
            value_added_process_completed=self.current_planned_value_added_process,
            process_executions=process_executions)
        for value_added_process_execution, _ in value_added_process_executions:
            actual_process_execution = value_added_process_execution.get_actual_process_execution()

            self.agent.current_work_order.update_period_by_actual(
                start_time=actual_process_execution.executed_start_time,
                end_time=actual_process_execution.executed_end_time,
                process_execution_id=actual_process_execution.identification,
                plan_process_execution_id=value_added_process_execution.identification)
        self.current_planned_value_added_process = None

    def get_value_added_process_execution(self) -> Optional[ProcessExecution]:
        value_added_process_executions = [process_execution
                                          for process_execution in self.agent.process_executions_order
                                          if isinstance(process_execution.process, ValueAddedProcess)]

        if len(value_added_process_executions) == 0:
            return None

        value_added_process_execution: ProcessExecution = value_added_process_executions[-1]
        value_added_process_execution_actual = value_added_process_execution.get_actual_process_execution()
        return value_added_process_execution_actual

    def get_next_value_added_processes(self) -> Optional[List[ValueAddedProcess]]:
        """
        The method is used to select the next ValueAddedProcess. Therefore, two ways are possible:
        Building up on the possible ValueAddedProcess, the function chooses the first ValueAddedProcess
        of the list "possible_value_added_processes" as the next ValueAddedProcess.
        The method is used to find the ValueAddedProcess to that is selected as next process to execute
        :return ValueAddedProcess
        """
        next_value_added_processes = None

        possible_value_added_processes = \
            self.agent.current_work_order.possible_value_added_processes
        if possible_value_added_processes:
            # ToDo: (later) logic should be extended in further versions in a way that
            #  the best value_added_process is chosen
            weights = self.get_weights(possible_value_added_processes)

            next_value_added_processes_chosen_permutation = choices(possible_value_added_processes, weights=weights)
            next_value_added_processes = [next_value_added_processes_chosen_permutation[0]]
            # if len(next_value_added_processes_chosen_permutation) > 1:
            #     next_value_added_processes.append(next_value_added_processes_chosen_permutation[-1])
            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       f" The process '{next_value_added_process.name}' was chosen")

        return next_value_added_processes

    def get_weights(self, possible_value_added_processes):
        """Calculate the weights"""

        process_material_supplies_needed = []
        for possible_value_added_process in possible_value_added_processes:
            # location_match = len([origin for
            #                       origin in possible_value_added_process.get_possible_origins()
            #                       if origin == self.current_location])

            material_supplies_needed = \
                int(sum([root_node.amount
                         for root_node in possible_value_added_process.transformation_controller.get_root_nodes()
                         if root_node.transformation_type_sub_part() and root_node.io_behaviour_exist()]))

            process_material_supplies_needed.append(material_supplies_needed)

        weights = [abs(material_supplies_needed - 1 * self.phase)  # ToDo: something went wrong
                   for material_supplies_needed in process_material_supplies_needed]

        span = max(weights) - min(weights) + 1 * 10 - 9
        weights = [((weight + 1 * 10 - 9) / span) + 1 for weight in weights]

        return weights

    async def _skip_negotiation_round(self, amount=1, until_end=False):
        """Skip negotiation_round
        first for part and support organization
        second for vap organization
        """
        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Skip {amount} negotiation rounds")
        # until_end = False  # ToDo: maybe a Schmaus problem???

        # if self.agent.current_order is not None:
        #     self.logging.info(f'Order ID {self.agent.current_order.identification} skipped {amount} rounds', extra={"obj_id": self.agent.name})
        # else:
        #     self.logging.info(f'Order ID {self.agent.current_order} skipped {amount} rounds', extra={"obj_id": self.agent.name})

        for i in range(amount):
            # sync
            round_, skipped_to_end = \
                await self.agent.agents.ready_for_next_round(agent_name=self.agent.name, until_end=until_end)

            if skipped_to_end:
                break

        return skipped_to_end

    def derive_preconditions_value_added_process(self, value_added_processes) -> dict[str: EntityType | None]:
        """
        Derive preconditions from the value_added_process
        Firstly it is checked if the main_part needs a SUPPORT through the production. If it is the case,
        the main_part is managed by the order agent.
        :param value_added_processes: the basis for the preconditions' derivation
        :return: a dict with preconditions for the value_added_process (main_part and support_transport_resource)
        """
        value_added_process = value_added_processes[0]
        value_added_process_preconditions = {"main_part_entity_type": None,
                                             "support_entity_type": None,
                                             "long_time_support_needed": None}
        self.value_added_processes_preconditions[value_added_process] = value_added_process_preconditions

        # if no main part chosen - search for the main_part of the process
        if value_added_process_preconditions["main_part_entity_type"]:
            return

        main_part_entity_type, support_entity_type, support_needed_long_time = \
            self._get_main_part_support(value_added_process)
        value_added_process_preconditions["main_part_entity_type"] = main_part_entity_type
        value_added_process_preconditions["support_entity_type"] = support_entity_type
        value_added_process_preconditions["long_time_support_needed"] = support_needed_long_time
        # if main_part found - the main_part should be organized - else no further process is needed

        # else:
        #     main_part_entity_type = value_added_process_preconditions["main_part_entity_type"]
        #     support_entity_type = value_added_process_preconditions["support_entity_type"]
        #     main_part_node = \
        #         [root_node for root_node in value_added_process.transformation_controller.get_root_nodes()
        #          if root_node.transformation_type_main_entity() and
        #          root_node.entity_type.identification == main_part_entity_type.identification]
        # if main_part_node:
        #     support_node = \
        #         [parent_node for parent_node in main_part_node[0].children[0].parents
        #          if parent_node.transformation_type_support()
        #          and parent_node.entity_type.identification == support_entity_type.identification]
        #     if not support_node:
        #         pass  # ToDo: (later) replan the support_entity_type
        # else:
        #     pass  # ToDo: (later) replan the main_part_entity_type

    def _get_main_part_support(self, value_added_process) -> tuple[EntityType | None, EntityType | None, bool]:
        """
        Determine the entity_type of the main_part and his support needed.
        Assumption: The main_part is determined at the beginning of the production process. This means that the part
        transformation nodes have logically a specific entity_type for the MAIN_PART (no super_entity_type).
        :return the entity_type of the main_part and his support as tuple or if no support needed two times none and
        a bool for long time support needed
        """
        # determine all main_parts and support root_nodes needed
        main_part_support_root_nodes = \
            [root_node for root_node in value_added_process.transformation_controller.get_root_nodes()
             if (root_node.transformation_type_main_entity() or root_node.transformation_type_support()) and
             not root_node.io_behaviour_created()]

        for main_part_or_support_root_node in main_part_support_root_nodes:
            # recheck if the child of a node has a main_part and support as parent/ they are connected
            # assumption: it is not possible that two main_parts have one children_node, also true for supports
            if not main_part_or_support_root_node.children:
                # destroy the main_part
                main_part_entity_type = main_part_or_support_root_node.entity_type
                return main_part_entity_type, None, False

            main_part_or_support_root_nodes = main_part_or_support_root_node.children[0].parents
            main_part_support_etn = [EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                     EntityTransformationNode.TransformationTypes.SUPPORT]

            main_part_support_pair = \
                [main_part_or_support_root_node
                 for main_part_or_support_root_node in main_part_or_support_root_nodes
                 if main_part_or_support_root_node.compare_transformation_type_self(main_part_support_etn)]

            if not main_part_support_pair:
                raise ValueError(f"Excel DEBUG needed (exception written for the bicycle world)")

            if len(main_part_support_pair) != 2:  # assumption only one main_part and one support
                if not len(main_part_support_pair) == 1:
                    continue
                if ((main_part_support_pair[0].transformation_type ==
                        EntityTransformationNode.TransformationTypes.MAIN_ENTITY) and
                        len(main_part_support_root_nodes) == 1):
                    main_part_entity_type = main_part_support_pair[0].entity_type

                    return main_part_entity_type, None, False
                continue

            # determine the main_part and transport entity_types
            main_part_support_entity_types = [main_part_support_node.entity_type
                                              for main_part_support_node in main_part_support_pair]
            if main_part_support_pair[0].transformation_type_support():
                support_entity_type, main_part_entity_type = main_part_support_entity_types
            else:
                main_part_entity_type, support_entity_type = main_part_support_entity_types

            support_needed_long_time = self._determine_longer_used_support(value_added_process, support_entity_type)

            # if not main_part_support_pair[0].children[0].children:
            #     # further process (for example something should be appended) is needed to be the main_part
            #     main_part_entity_type = None

            return main_part_entity_type, support_entity_type, support_needed_long_time

        return None, None, False

    def _determine_longer_used_support(self, value_added_process, support_entity_type) -> bool:
        # check if the support entity_type is found in the successor processes -
        # longer needed preconditions are managed by the order agent respectively support that are
        # only needed for the current process are currently managed by providers

        possible_next_value_added_processes = self.agent.current_work_order.possible_value_added_processes
        if possible_next_value_added_processes:
            successors = possible_next_value_added_processes
        else:
            successors = value_added_process.successors

        support_nodes = []
        for successor in successors:
            support_nodes_batch = successor.get_support_entity_types(support_entity_type)
            if support_nodes_batch:
                support_nodes.extend(support_nodes_batch)

        # Note: not all successors can/ must be chosen, because they are representative for all features
        output_entity_types = \
            [entity_type for entity_type, amount in value_added_process.get_possible_output_entity_types()]
        longer_used_supports = [support_node for support_node in support_nodes
                                for parent_node in support_node.children[0].parents
                                for output_entity_type in output_entity_types
                                if parent_node.entity_type.check_entity_type_match(output_entity_type)]

        if len(longer_used_supports) > 0:
            return True
        else:
            return False

    async def _organize(self, value_added_processes, old_value_added_process_execution):
        """
        The organization is split in three steps:
        1. Loading: here the product is loaded to a support or a picking box is started to flow
        2. VAP: the value added process is organized
        3. transport: the transport to the location where the process executed is organized
        If one of the steps is not needed it is skipped elif not successfully achieved the whole process is aborted
        """
        # print(get_debug_str(self.agent.name, self.__class__.__name__) + " Start organization")
        self.logging.info(f"Order ID: {self.agent.current_order.identification}, Organization started",
                          extra={"obj_id": self.agent.name})
        requested_times = {}
        value_added_process = value_added_processes[0]
        preconditions = self.value_added_processes_preconditions[value_added_process]
        main_part_entity_type = preconditions["main_part_entity_type"]
        support_entity_type = preconditions["support_entity_type"]

        if self.reset_current_resource_binding:
            if self.current_resource_binding is not None:
                await self._reset_resource_binding()

        last_process_executions, main_organization_needed = \
            self._determine_main_organization_needed(old_value_added_process_execution, main_part_entity_type,
                                                     support_entity_type)

        if (main_part_entity_type or support_entity_type) and main_organization_needed:
            # print(get_debug_str(self.agent.name, self.__class__.__name__) + " Main Loading")
            self.logging.info(f"Order ID: {self.agent.current_order.identification}, Main Loading",
                              extra={"obj_id": self.agent.name})
            # Main loading
            loading_successful, process_executions_loading_with_provider, planned_process_executions = \
                await self._organize_main_loading(value_added_process=value_added_process,
                                                  main_part_entity_type=main_part_entity_type,
                                                  support_entity_type=support_entity_type)

            if not loading_successful:
                await self._skip_negotiation_round(amount=2, until_end=True)
                return loading_successful, requested_times
            else:
                self._main_organization_needed = {}

            resource_binding, last_process_execution = \
                self._get_resource_binding(planned_process_executions, support_entity_type)
            if last_process_execution:
                self.set_current_location_origin(last_process_execution)
            self.value_added_process_call_for_proposals = None
        else:
            planned_process_executions = []
            process_executions_loading_with_provider = []
            _, last_process_execution = self._get_process_executions_order()
            await self._skip_negotiation_round(amount=1)
            resource_binding = None
        process_executions_with_provider = process_executions_loading_with_provider

        if last_process_execution:
            start_time_transport = last_process_execution.executed_end_time
        else:
            start_time_transport = self.current_time64.item()

        # if self.support_resource is not None:
        #     print(self.support_resource.name, start_time_transport)
        # VAP
        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" VAP")
        self.logging.info(
            f'Order ID: {self.agent.current_order.identification}, VAP {value_added_process.identification}',
            extra={"obj_id": self.agent.name})
        (vap_successful, process_execution_batch_with_provider, value_added_process_execution,
         planned_process_executions_vap, accepted_time_period) = \
            await self._organize_value_added_process(
                value_added_processes=value_added_processes, last_process_execution=last_process_execution,
                last_value_added_process_execution=old_value_added_process_execution,
                support_entity_type=support_entity_type, main_part_entity_type=main_part_entity_type)

        print(get_debug_str(self.agent.name, self.__class__.__name__) + f" VAP {vap_successful}")
        if value_added_process_execution is not None:
            self.logging.info(
                f'Order ID: {self.agent.current_order.identification}, VAP:{value_added_process_execution.process.name} is {vap_successful}, PE:{value_added_process_execution.identification} ',
                extra={"obj_id": self.agent.name})
        else:
            self.logging.info(
                f'Order ID: {self.agent.current_order.identification}, VAP: {value_added_process_execution} is {vap_successful}',
                extra={"obj_id": self.agent.name})
        if not vap_successful:
            requested_times["VAP"] = accepted_time_period
            await self._cancel_process_execution_batch(
                process_executions_with_provider=process_executions_with_provider)
            if main_organization_needed:
                self.current_location = None

            self.phase += 1
            await self._skip_negotiation_round(amount=1, until_end=True)
            return vap_successful, requested_times

        else:
            self.value_added_process_call_for_proposals = None

        planned_process_executions += planned_process_executions_vap
        process_executions_with_provider.extend(process_execution_batch_with_provider)
        # Transport
        destination = self.get_destination(value_added_process_execution)
        end_time_transport = value_added_process_execution.executed_start_time
        self.set_support_resource(value_added_process_execution, value_added_process,
                                  planned_process_executions, support_entity_type)
        main_process_execution = last_process_execution
        if Responsibility.MAIN_ENTITY_TRANSPORT in self.agent.responsibilities:
            transport_successful, process_execution_transport_with_provider, accepted_time_period = \
                await self._organize_main_transport_process(
                    destination=destination, main_process_execution=main_process_execution,
                    main_part_entity_type=main_part_entity_type, support_entity_type=support_entity_type,
                    start_time=start_time_transport, end_time=end_time_transport,
                    planned_process_executions=planned_process_executions)
        else:
            await self._skip_negotiation_round(amount=1)
            process_execution_transport_with_provider = []
            transport_successful = True

        print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Transport {transport_successful}")
        self.logging.info(f'Order ID: {self.agent.current_order.identification}, Transport {transport_successful}',
                          extra={"obj_id": self.agent.name})
        if not transport_successful:
            requested_times["TRANSPORT"] = accepted_time_period
            await self._cancel_process_execution_batch(
                process_executions_with_provider=process_executions_with_provider)

            self.phase += 1
            return transport_successful, requested_times

        # print(f"Value Added Process processed: {value_added_process.name}")
        process_executions_with_provider.extend(process_execution_transport_with_provider)
        if resource_binding is not None:
            self.current_resource_binding = resource_binding

        self.set_current_location_destination(value_added_process_execution)
        self.phase = 0
        self.current_planned_value_added_process = value_added_process
        await self._handle_process_executions_batch(value_added_process=value_added_process,
                                                    value_added_process_execution=value_added_process_execution,
                                                    process_executions_with_provider=process_executions_with_provider,
                                                    resource_binding=resource_binding,
                                                    main_part_entity_type=main_part_entity_type)
        # for process_execution, _ in process_executions_with_provider:
        #     print("Order Process Name:", process_execution.identification, process_execution.get_process_name())
        #     print(process_execution.executed_start_time, process_execution.executed_end_time)

        self._set_order_product(value_added_process_execution)
        self.logging.info(f'Order ID: {self.agent.current_order.identification}, Organization ended',
                          extra={"obj_id": self.agent.name})
        return True, requested_times

    def _get_process_executions_order(self, planned_process_executions=None):
        """Get the last process_execution"""

        if planned_process_executions is not None:
            if isinstance(planned_process_executions, list):
                process_execution_lst = planned_process_executions
            else:
                process_execution_lst = [planned_process_executions]
        else:
            process_execution_lst = []

        # sort them
        process_executions_already_planned = self.agent.process_executions_order
        if len(process_execution_lst) > 0:
            process_execution_lst = _get_sorted_process_executions(process_execution_lst)
        process_executions_order = process_executions_already_planned + process_execution_lst

        if not process_executions_order:
            return [], None
        last_process_execution = process_executions_order[-1]

        # the ACTUAL PE has the newer information
        last_process_execution_actual = last_process_execution.get_actual_process_execution()
        if last_process_execution_actual is not None:
            last_process_execution = last_process_execution_actual

        return process_executions_order, last_process_execution

    def set_current_location_origin(self, process_execution):
        """Origin should be the standard"""
        if isinstance(process_execution.origin, StationaryResource):
            self.current_location = process_execution.origin
        elif isinstance(process_execution.destination, StationaryResource):
            self.current_location = process_execution.destination

    def set_current_location_destination(self, process_execution):
        """Destination should be the standard"""
        if isinstance(process_execution.destination, StationaryResource):
            self.current_location = process_execution.destination
        elif isinstance(process_execution.origin, StationaryResource):
            self.current_location = process_execution.origin

    async def _handle_orders_already_started_before(self):
        """handle already started orders"""

        order = self.agent.current_order

        await self._bind_resource_order_already_started_before(order)

        process_executions_completed = order.get_process_executions()

        if process_executions_completed:
            process_executions_completed = _get_sorted_process_executions(process_executions_completed)
            self._old_value_added_process_execution = process_executions_completed[-1]
            self.agent.process_executions_finished.append(self._old_value_added_process_execution)
            self.agent.process_executions_order.extend(process_executions_completed)

        if self._old_value_added_process_execution:
            self.set_current_location_destination(self._old_value_added_process_execution)

    async def _bind_resource_order_already_started_before(self, order):
        """
        Set a resource binding if required to ensure that no other order take the resource already linked with the order
        """
        products = order.products
        if not products:
            return

        product = products[0]  # ToDo: maybe all products should be considered
        situated_in = product.situated_in
        while situated_in:
            if not isinstance(situated_in, Storage):
                break

            situated_in = situated_in.situated_in
        # print("situated_in", situated_in)
        if not situated_in:
            return

        if not isinstance(situated_in, NonStationaryResource):
            return

        self.support_resource = situated_in

        self.current_resource_binding = (self.support_resource, order)
        print(f"BIND RESOURCE: {self.support_resource.name}", order.identifier, order.identification)
        await self.bind_resource(self.current_resource_binding)

    def _determine_main_organization_needed(self, old_value_added_process_execution, main_part_entity_type,
                                            support_entity_type):
        """Determine if the main organization is needed"""

        if (old_value_added_process_execution, main_part_entity_type, support_entity_type) \
                in self._main_organization_needed:
            last_process_executions, organization_needed = \
                self._main_organization_needed[(old_value_added_process_execution, main_part_entity_type,
                                                support_entity_type)]

            return last_process_executions, organization_needed

        last_process_executions = []

        # if old value_added_process_executions exist, it can be the case that the already chosen part can be used for
        # the next process_executions_components
        if old_value_added_process_execution is None:  # ToDo: DEBUGGING needed
            organization_needed = True
            self._main_organization_needed = {(old_value_added_process_execution, main_part_entity_type,
                                               support_entity_type):
                                                  (last_process_executions, organization_needed)}

        if main_part_entity_type is not None:
            process_executions_main = \
                {process_execution.executed_end_time: process_execution
                 for process_execution in self.agent.process_executions_order
                 for part in process_execution.get_parts()
                 if part.entity_type.check_entity_type_match(main_part_entity_type)}

            if not process_executions_main:
                organization_needed = True
                self._main_organization_needed = {(old_value_added_process_execution, main_part_entity_type,
                                                   support_entity_type):
                                                      (last_process_executions, organization_needed)}
                return last_process_executions, organization_needed

            process_executions_main_max = max(process_executions_main.keys())

            last_process_execution_main = process_executions_main[process_executions_main_max]
            last_process_executions = [last_process_execution_main]

        elif support_entity_type is not None:
            process_executions_main = \
                {process_execution.connected_process_execution.executed_end_time:
                     process_execution.connected_process_execution
                 for process_execution in self.agent.process_executions_order
                 for resource in process_execution.connected_process_execution.get_resources()
                 if resource.entity_type.check_entity_type_match(support_entity_type)}

            process_executions_main_max = max(process_executions_main.keys())
            last_process_execution_main = process_executions_main[process_executions_main_max]
            last_process_executions = [last_process_execution_main]

        else:
            last_process_executions = self.agent.process_executions_order.copy()

        organization_needed = False
        self._main_organization_needed = \
            {(old_value_added_process_execution, main_part_entity_type, support_entity_type):
                 (last_process_executions, organization_needed)}
        return last_process_executions, organization_needed

    def _get_resource_binding(self, planned_process_executions, support_entity_type):
        _, last_process_execution = \
            self._get_process_executions_order(planned_process_executions=planned_process_executions)
        if support_entity_type is None:
            return None, last_process_execution

        resource_binding = (last_process_execution.get_support_resource(support_entity_type), self.agent.current_order)

        return resource_binding, last_process_execution

    async def _organize_main_loading(self, value_added_process, main_part_entity_type, support_entity_type,
                                     possible_origins=[]):
        await_callback, requests = \
            await self._organize_main_loading_request(value_added_process=value_added_process,
                                                      main_part_entity_type=main_part_entity_type,
                                                      support_entity_type=support_entity_type,
                                                      possible_origins=possible_origins)

        if await_callback:
            loading_successful, process_executions_loading_with_provider, planned_process_executions = \
                await self._organize_main_loading_response(await_callback, requests)

        else:
            await self._skip_negotiation_round(amount=1)
            loading_successful = True
            process_executions_loading_with_provider = []
            planned_process_executions = []

        return loading_successful, process_executions_loading_with_provider, planned_process_executions

    async def _organize_main_loading_request(self, value_added_process, main_part_entity_type, support_entity_type,
                                             possible_origins=[]):
        """Organize the main part (can also be a resource - in the case of picking box)"""

        # organize main_part entity_type
        loading_process: Process = self._get_loading_processes(value_added_process,
                                                               main_part_entity_type, support_entity_type,
                                                               possible_origins)

        if not loading_process:
            return None, []

        requests = [loading_process]
        parts_involved = []
        if self.agent.current_order.products:
            for product in self.agent.current_order.products:  # ToDo: maybe a selection is required
                if loading_process.check_possible_input_part(possible_input_part=product):
                    parts_involved = [(product,)]
                    break

        process_execution = \
            ProcessExecution(event_type=ProcessExecution.EventTypes.PLAN, order=self.agent.current_work_order.order,
                             process=loading_process, parts_involved=parts_involved, resulting_quality=1,
                             source_application=self.agent.source_application)

        if self.support_resource is not None:
            organize_transport_access = False
            support = self.support_resource
        else:
            organize_transport_access = True
            support = support_entity_type
        long_time_reservation = {support: (self.determine_expected_transport_time(),
                                           organize_transport_access)}

        accepted_time_periods = np.array([[self.current_time64, np.datetime64("NaT")]],
                                         dtype='datetime64[s]')
        expected_process_execution_time = process_execution.get_max_process_time(distance=0)
        preference = self.agent.current_work_order_preference.get_process_execution_preference(
            expected_time_slot_duration=expected_process_execution_time, reference_objects=[process_execution],
            accepted_time_periods=accepted_time_periods)

        process_call_for_proposal = self._get_process_call_for_proposal(process_execution, preference,
                                                                        long_time_reservation)

        if main_part_entity_type and main_part_entity_type in self.agent.entity_provider:
            providers_entity_type = main_part_entity_type
        else:
            providers_entity_type = support_entity_type

        providers = self.agent.entity_provider[providers_entity_type]
        await_callback = await self._request_process_call_for_proposal(process_call_for_proposal, providers)

        return await_callback, requests

    def _get_loading_processes(self, value_added_process, main_part_entity_type, support_entity_type, possible_origins):
        """Organize the loading processes and the providers"""

        loading_processes = self._determine_loading_processes(main_part_entity_type, support_entity_type,
                                                              possible_origins)

        if len(loading_processes) != 1:
            loading_processes = self._choose_loading_process(value_added_process, loading_processes,
                                                             support_entity_type, main_part_entity_type)
        if not loading_processes:
            return None

        loading_process = loading_processes[0]

        return loading_process

    def _determine_loading_processes(self, main_part_entity_type, support_entity_type, possible_origins):
        """
        Determine the main loading processes -
        can be for a support or a part that is loaded and transported through the shop floor.
        """

        if main_part_entity_type:
            if not possible_origins and main_part_entity_type in self.agent.entity_provider:
                possible_origins = self.agent.entity_provider[main_part_entity_type]

            loading_processes = \
                [loading_process
                 for loading_process in self.agent.processes["loading_processes"]
                 if self._check_process_usable_as_loading_main(loading_process, main_part_entity_type,
                                                               possible_origins)]

        else:
            support_provider_resources = self.agent.entity_provider[support_entity_type]
            loading_processes = \
                [loading_process
                 for loading_process in self.agent.processes["loading_processes"]
                 if self._check_process_usable_as_loading_support(support_provider_resources, loading_process,
                                                                  support_entity_type)]

        return loading_processes

    def _choose_loading_process(self, value_added_process, loading_processes,
                                support_entity_type, main_part_entity_type):
        # check which process is before the value_added_process

        possible_origins = value_added_process.get_possible_origins()
        loading_processes_to_remove = []
        for loading_process in loading_processes:
            loading_process_possible = False

            for possible_origin_loading in loading_process.get_possible_origins():
                for possible_origin_value_creation in possible_origins:
                    if possible_origin_loading == possible_origin_value_creation:
                        loading_process_possible = True
                        break

                    transport_processes = (
                        self.agent.routing_service.get_transit_processes(
                            origin=possible_origin_loading, destination=possible_origin_value_creation,
                            support_entity_type=support_entity_type,
                            entity_entity_type=main_part_entity_type,
                            transfers=True, solution_required=False))

                    if len(transport_processes) > 0:
                        loading_process_possible = True
                        break

            if not loading_process_possible:
                loading_processes_to_remove.append(loading_process)

        # select the loading processes that are before the value added process
        loading_processes = list(set(loading_processes).difference(set(loading_processes_to_remove)))

        return loading_processes

    def _check_process_usable_as_loading_main(self, loading_process, main_part_entity_type, possible_origins):
        """Check if the loading_process is usable as loading_process"""
        possible_origins_loading = loading_process.get_possible_origins()
        for possible_origin in possible_origins_loading:
            if possible_origin in possible_origins or not possible_origins:
                matching_entity_types = \
                    [input_entity_type for input_entity_type in loading_process.get_input_entity_types_set()
                     if input_entity_type.check_entity_type_match(main_part_entity_type)]
                if matching_entity_types:
                    return loading_process

        return None

    def _check_process_usable_as_loading_support(self, support_provider_resources, loading_process,
                                                 support_entity_type):
        """Check if the loading_process is usable as loading_process"""
        possible_origins = loading_process.get_possible_origins()
        for support_provider_resource in support_provider_resources:
            for possible_origin in possible_origins or not possible_origins:
                if possible_origin.identification != support_provider_resource.identification:
                    continue
                if loading_process.get_necessary_input_entity_types()[0][0].check_entity_type_match(
                        support_entity_type):
                    return loading_process

        return None

    def _get_process_call_for_proposal(self, process_execution, preference, long_time_reservation):
        """Call for proposal (Process)"""

        process_call_for_proposal = \
            ProcessCallForProposal(reference_cfp=None, sender_name=self.agent.name,
                                   request_type=CallForProposal.RequestType.FLEXIBLE,
                                   client_object=self.agent.current_order, order=self.agent.current_order,
                                   issue_id=ProcessCallForProposal.next_id, process_execution=process_execution,
                                   preference=preference, long_time_reservation=long_time_reservation)
        process_call_for_proposal.issue_id = process_call_for_proposal.identification
        return process_call_for_proposal

    async def _request_process_call_for_proposal(self, process_call_for_proposal, providers):
        """Call for proposal (Process)"""

        process_call_for_proposal_ids = [process_call_for_proposal.identification]

        round_, skipped_to_end = await self.agent.agents.ready_for_next_round(agent_name=self.agent.name)
        possible_proposal_providers = self.agent.NegotiationBehaviour.convert_providers(providers)
        self.agent.agents.set_await_proposal(possible_proposal_providers=possible_proposal_providers, round_=round_)

        await self.agent.NegotiationBehaviour.call_for_proposal(process_call_for_proposal, providers)
        await_callback = \
            asyncio.create_task(self.agent.NegotiationBehaviour.await_callback(process_call_for_proposal_ids))

        return await_callback

    async def _organize_main_loading_response(self, await_callback, requests):
        """Organize the response to a main loading request"""

        process_executions, best_time_slot_value_proposals = \
            await self._get_process_executions(await_callback, requests)

        if best_time_slot_value_proposals:
            loading_successful = True
            planned_process_executions = process_executions

            for process_execution in process_executions:
                if not process_execution.executed_start_time:
                    raise Exception

        else:
            loading_successful = False
            planned_process_executions = []
        # ToDo: consider provider if needed
        process_executions_with_provider = append_provider_to_process_executions(process_executions, None)

        return loading_successful, process_executions_with_provider, planned_process_executions

    async def _organize_value_added_process(self, value_added_processes, last_process_execution,
                                            last_value_added_process_execution, support_entity_type,
                                            main_part_entity_type):
        """Organize the value added process by requesting by the other agent_control"""
        # print(self.agent.name, [vap.name for vap in value_added_processes])
        requests, await_callbacks, accepted_time_period, process_call_for_proposals = \
            await self._organize_value_added_process_request(
                value_added_processes=value_added_processes, last_process_execution=last_process_execution,
                last_value_added_process_execution=last_value_added_process_execution,
                support_entity_type=support_entity_type, main_part_entity_type=main_part_entity_type)

        vap_successful = False

        if await_callbacks:
            (vap_successful, process_execution_batch_with_provider, value_added_process_execution,
             planned_process_executions) = \
                await self._organize_value_added_process_response(await_callbacks=await_callbacks,
                                                                  value_added_processes=value_added_processes,
                                                                  requests=requests)

        else:
            process_execution_batch_with_provider = []
            planned_process_executions = []
            value_added_process_execution = None

        return (vap_successful, process_execution_batch_with_provider, value_added_process_execution,
                planned_process_executions, accepted_time_period)

    async def _organize_value_added_process_request(self, value_added_processes, last_process_execution,
                                                    last_value_added_process_execution,
                                                    support_entity_type, main_part_entity_type) \
            -> [tuple, list[ValueAddedProcess], list[asyncio.Task], np.array, list[ProcessCallForProposal]]:
        """organize value_added_process_executions"""

        process_call_for_proposals, providers_dict, accepted_time_period = \
            self._get_value_added_process_call_for_proposals(value_added_processes, last_process_execution,
                                                             last_value_added_process_execution,
                                                             support_entity_type, main_part_entity_type)

        await_callbacks = await self._request_process_call_for_proposals(process_call_for_proposals, providers_dict)

        # print(get_debug_str(self.agent.name, self.__class__.__name__), "VAP Request",
        #       [vap.name for vap in value_added_processes],
        #       [process_call_for_proposal.process_execution.process.name
        #        for process_call_for_proposal in process_call_for_proposals])

        # print("VAP cfps: ", self.agent.name, len(process_call_for_proposals))
        return copy(value_added_processes), await_callbacks, accepted_time_period, process_call_for_proposals

    def _get_value_added_process_call_for_proposals(self, value_added_processes, last_process_execution,
                                                    last_value_added_process_execution,
                                                    support_entity_type, main_part_entity_type):

        take_cache = False
        if self.value_added_process_call_for_proposals is not None:
            process_call_for_proposals, providers_dict, last_process_execution, accepted_time_period = (
                self.value_added_process_call_for_proposals)

            # Note: this is no complete check!
            if process_call_for_proposals[0].process_execution.process == value_added_processes[0]:
                take_cache = True

        if not take_cache:
            # from scratch
            process_call_for_proposals, providers_dict, accepted_time_period = \
                self._get_value_added_process_call_for_proposals_from_scratch(value_added_processes,
                                                                              last_process_execution,
                                                                              last_value_added_process_execution,
                                                                              support_entity_type,
                                                                              main_part_entity_type)
            self.value_added_process_call_for_proposals = \
                process_call_for_proposals, providers_dict, last_process_execution, accepted_time_period

        else:
            # from the round before
            process_call_for_proposals, providers_dict, last_process_execution, accepted_time_period = \
                self.value_added_process_call_for_proposals

            transport_required = support_entity_type or main_part_entity_type
            preference, accepted_time_period = \
                self._get_preference_value_added_process(process_call_for_proposals[0].process_execution,
                                                         last_process_execution,
                                                         value_added_processes[0], transport_required)

            self.value_added_process_call_for_proposals = \
                process_call_for_proposals, providers_dict, last_process_execution, accepted_time_period

            for idx, process_call_for_proposal in enumerate(process_call_for_proposals):
                if len(process_call_for_proposals) < idx + 1:
                    preference = preference.copy()
                process_call_for_proposal.preference = preference

        # for process_call_for_proposal in process_call_for_proposals:
        #     print("ID: ", process_call_for_proposal.process_execution.identification)

        return process_call_for_proposals, providers_dict, accepted_time_period

    def _get_value_added_process_call_for_proposals_from_scratch(self, value_added_processes,
                                                                 last_process_execution,
                                                                 last_value_added_process_execution,
                                                                 support_entity_type, main_part_entity_type):
        """Create value added process call for proposal"""

        process_call_for_proposals = []
        providers_dict = {}
        main_part_process_execution = last_process_execution
        if last_value_added_process_execution is not None:
            main_part_process_execution = last_value_added_process_execution
        else:
            last_value_added_process_execution = self.get_value_added_process_execution()
            if last_value_added_process_execution is not None:
                main_part_process_execution = last_value_added_process_execution

        for value_added_process in value_added_processes:
            parts_involved, long_time_reservation = \
                self._get_entities_value_added_process(value_added_process, support_entity_type, main_part_entity_type,
                                                       main_part_process_execution)

            process_execution = \
                ProcessExecution(event_type=ProcessExecution.EventTypes.PLAN, process=value_added_process,
                                 parts_involved=parts_involved, resulting_quality=1, resources_used=[],
                                 main_resource=None, order=self.agent.current_work_order.order,
                                 source_application=self.agent.source_application)
            transport_required = support_entity_type or main_part_entity_type
            preference, accepted_time_period = \
                self._get_preference_value_added_process(process_execution, last_process_execution, value_added_process,
                                                         transport_required)
            main_resource = value_added_process.get_resource_groups()[0].main_resources[0]
            try:
                providers = [self.agent.address_book[main_resource]]
            except KeyError:
                raise Exception(main_resource.name, self.agent.name)

            process_call_for_proposal = self._get_process_call_for_proposal(process_execution, preference,
                                                                            long_time_reservation)
            process_call_for_proposals.append(process_call_for_proposal)

            providers_dict[process_call_for_proposal] = providers

        return process_call_for_proposals, providers_dict, accepted_time_period

    async def _request_process_call_for_proposals(self, process_call_for_proposals, providers):

        round_, skipped_to_end = await self.agent.agents.ready_for_next_round(agent_name=self.agent.name)
        providers_all = [provider
                         for provider_lst in list(providers.values())
                         for provider in provider_lst]
        possible_proposal_providers = providers_all
        self.agent.agents.set_await_proposal(possible_proposal_providers=possible_proposal_providers, round_=round_)

        await_callbacks = []
        for process_call_for_proposal in process_call_for_proposals:
            process_call_for_proposal_ids = [process_call_for_proposal.identification]
            await self.agent.NegotiationBehaviour.call_for_proposal(process_call_for_proposal,
                                                                    providers[process_call_for_proposal])
            await_callback = \
                asyncio.create_task(self.agent.NegotiationBehaviour.await_callback(process_call_for_proposal_ids))

            await_callbacks.append(await_callback)

        return await_callbacks

    def _get_entities_value_added_process(self, value_added_process, support_entity_type, main_part_entity_type,
                                          main_part_process_execution):
        """Determine the entities usable from the process_executions before"""

        if support_entity_type:
            organize_transport_access = False
            long_time_reservation = \
                {main_part_process_execution.get_support_resource(support_entity_type):
                     (self.determine_expected_transport_time(),
                      organize_transport_access)}
        else:
            long_time_reservation = {}
            self.reset_current_resource_binding = True
        print("Set Support", self.agent.current_order.identification, value_added_process.name,
              support_entity_type.name if support_entity_type is not None else None)

        if self.agent.current_order.products:
            available_parts = [part
                               for part in self.agent.current_order.products
                               if part.entity_type.check_entity_type_match(main_part_entity_type)]
            possible_parts = value_added_process.choose_needed_parts_involved(available_parts=available_parts)

            # not already assembled - choose only the product
            parts_involved = [part_tuple
                              for part_tuple in possible_parts]

        elif main_part_entity_type:
            if main_part_process_execution is None:
                parts_involved = []

                return parts_involved, long_time_reservation

            actual_main_part_process_execution = main_part_process_execution.get_actual_process_execution()

            if actual_main_part_process_execution is not None:
                # take the output parts
                available_parts_involved = actual_main_part_process_execution.parts_involved
                if available_parts_involved:
                    # problem if creation and processed part in the same transformation process
                    # entity_transformation node should be available
                    available_parts = [part_tuple[0]
                                       for part_tuple in available_parts_involved
                                       if len(part_tuple) == 2]

                else:
                    plan_main_part_process_execution = main_part_process_execution.get_plan_process_execution()
                    available_parts = plan_main_part_process_execution.get_parts()

            else:
                plan_main_part_process_execution = main_part_process_execution.get_plan_process_execution()
                available_parts = plan_main_part_process_execution.get_parts()

            available_parts = [part for part in available_parts
                               if part.entity_type.check_entity_type_match(main_part_entity_type)]
            possible_parts = value_added_process.choose_needed_parts_involved(available_parts=available_parts)

            # not already assembled - choose only the product
            parts_involved = [part_tuple
                              for part_tuple in possible_parts
                              if part_tuple[0].part_of is None]

        else:
            parts_involved = []

        return parts_involved.copy(), long_time_reservation

    def get_accepted_time_periods_vap(self, last_process_execution, value_added_process, transport_required):

        start_time_period, lead_time = \
            self.determine_start_time_value_added_process(last_process_execution,
                                                          value_added_process=value_added_process,
                                                          transport_required=transport_required)

        start_time_period64 = np.datetime64(start_time_period.replace(microsecond=0))

        # print("VAP requested before", self._requested_times_before)

        start_time_period64 = np.maximum(self.current_time64, start_time_period64)

        if "VAP" not in self._requested_times_before:
            extension = None

        elif self._requested_times_before["VAP"].size == 0:
            extension = None

        else:
            extension = (self._requested_times_before["VAP"][1] - self._requested_times_before["VAP"][0] +
                         np.timedelta64(450, "s")).item().seconds

        accepted_time_periods = np.array([[start_time_period64, np.datetime64("NaT")]],
                                         dtype='datetime64[s]')

        return accepted_time_periods, extension

    def _get_preference_value_added_process(self, process_execution, last_process_execution, value_added_process,
                                            transport_required):
        """Determine the preference for the value_added_process"""
        start_funtion = datetime.now()
        if self.last_pe == value_added_process.identification:
            self.negotiation_failed += 1
        self.last_pe = value_added_process.identification
        self.amount_negotiation += 1

        accepted_time_periods, extension = \
            self.get_accepted_time_periods_vap(last_process_execution, value_added_process, transport_required)

        distance = process_execution.get_distance()
        time_slot_duration = int(np.ceil(round(process_execution.get_max_process_time(distance=distance), 1)))
        preference = self.agent.current_work_order_preference.get_process_execution_preference(
            expected_time_slot_duration=time_slot_duration, reference_objects=process_execution,
            accepted_time_periods=accepted_time_periods, extension=extension)

        accepted_time_period = preference.accepted_time_periods[0]

        return preference, accepted_time_period

    async def _get_process_executions(self, await_callback, requests):
        """Wait until the proposals are available and choose the best proposal
        Based on that the provider agent will deliver process_executions"""

        proposals = await await_callback

        if not proposals:  # negotiation failed
            process_executions = []
            best_time_slot_value_proposal = None
            return process_executions, best_time_slot_value_proposal

        best_time_slot_value_proposals, proposals_to_refuse = _choose_best_proposals(proposals, requests,
                                                                                     agent_name=self.agent.name)
        messages_awaited_size = len(best_time_slot_value_proposals)
        # print("Accepted:", messages_awaited_size, len(proposals_to_refuse), len(requests))
        # print("ACCEPTED FROM ORDER...", messages_awaited_size)
        await self._send_proposals_respond(proposals_to_propose=best_time_slot_value_proposals,
                                           proposals_to_refuse=proposals_to_refuse)

        # print(get_debug_str(self.agent.name, self.__class__.__name__), f" Start hearing {messages_awaited_size}")
        msg_process_executions_lst = []
        # print(self.agent.name, messages_awaited_size, len(proposals_to_refuse))
        for i in range(messages_awaited_size):
            msg_process_executions_received = \
                await self.agent.receive_msg(self, timeout=10,
                                             metadata_conditions={"ontology": "PROPOSAL"})

            if msg_process_executions_received is not None:
                msg_process_executions_lst.append(msg_process_executions_received)

        # print(get_debug_str(self.agent.name, self.__class__.__name__), msg_process_executions_lst)

        if msg_process_executions_lst:
            process_executions = []
            for msg_process_executions in msg_process_executions_lst:
                msg_content, msg_sender, msg_ontology, msg_performative = msg_process_executions
                process_executions.extend(msg_content)

            process_executions = _get_sorted_process_executions(process_executions)

            # for process_execution in process_executions:
            #     print(f"Process: {process_execution.process.name:40} {process_execution.identification}: \n"
            #           f"{process_execution.main_resource.name:40}"
            #           f"{process_execution.executed_start_time.time()} - {process_execution.executed_end_time.time()}")

            return process_executions, best_time_slot_value_proposals

        return [], best_time_slot_value_proposals

    async def _organize_value_added_process_response(self, await_callbacks, value_added_processes, requests):

        process_executions = []

        for await_callback in await_callbacks:
            process_executions, best_time_slot_value_proposals = \
                await self._get_process_executions(await_callback, requests)

        # if len(process_executions) > 1:
        #     raise Exception

        value_added_process = value_added_processes[0]

        if best_time_slot_value_proposals:

            successful = True
            process_executions_to_block = \
                [process_execution for process_execution in process_executions
                 if process_execution.process.identification == value_added_process.identification]

            if not process_executions_to_block:
                print("R1")
                raise Exception("The negotiation process failed")

            process_execution_to_block = process_executions_to_block[0]

            # ToDo: consider provider if needed
            process_executions_with_provider = append_provider_to_process_executions(process_executions, None)

        else:
            successful = False
            process_executions_with_provider = []
            process_execution_to_block = None

        return successful, process_executions_with_provider, process_execution_to_block, process_executions

    def determine_start_time_value_added_process(self, last_process_execution: ProcessExecution, value_added_process,
                                                 transport_required):
        """
        Determine the start_time for the value_added_process with consideration of earlier processes
        :param value_added_process: the value_added_process done next
        :return: the possible start_time of the value_added_process
        """
        if not transport_required:
            return self.current_time64.item(), timedelta(0)
        alternative_transport_times = self._get_transport_time_alternatives(value_added_process, last_process_execution)
        if not alternative_transport_times:
            return self.current_time64.item(), timedelta(0)
        max_transport_time = max(alternative_transport_times)

        lead_time = timedelta(seconds=int(np.ceil(round(max_transport_time, 1))))
        executed_end_time = self.current_time64.item()
        if last_process_execution:
            if executed_end_time < last_process_execution.executed_end_time:
                executed_end_time = last_process_execution.executed_end_time

        start_time = executed_end_time + lead_time

        return start_time, lead_time

    def _get_transport_time_alternatives(self, value_added_process, last_process_execution):
        """Get the transport time for alternatives (different destinations)"""

        origin = self._get_transport_origin(value_added_process, last_process_execution)
        if not origin:
            alternative_transport_times = []
            return alternative_transport_times
        possible_destinations = value_added_process.get_possible_origins()
        # restocking processes are not considered
        if isinstance(possible_destinations[0], NonStationaryResource):
            possible_destinations = value_added_process.get_possible_destinations()

        alternative_transport_times = []
        for possible_destination in possible_destinations:
            # print("Origin:",  last_process_execution.origin, last_process_execution.process.name,
            # possible_destination)

            if origin.identification == possible_destination.identification:
                alternative_transport_times.append(0)
                continue

            if last_process_execution:
                # determine an entity_type that is used to ensure that the right process is chosen
                # e.g. an employee or an agv can transport things or change their position -
                # maybe for both exist a process
                support_entity_type = last_process_execution.get_support_entity_type()
                if not support_entity_type:
                    support_entity_type = None
                    # support_entity_type = last_process_execution.get_main_entity_entity_type()

                # restocking processes are not considered
                if isinstance(origin, NonStationaryResource | ConveyorBelt):
                    origin = last_process_execution.origin
            else:
                support_entity_type = None

            transport_transfer_processes = \
                self._get_transport_process_chain(origin, possible_destination, support_entity_type)

            transport_time = \
                sum([np.ceil(round(process_dict["process"].get_estimated_process_lead_time(
                    origin=process_dict["origin"],
                    destination=process_dict["destination"]), 1))
                    for process_dict in transport_transfer_processes])

            alternative_transport_times.append(int(transport_time))

        return alternative_transport_times

    def _get_transport_origin(self, value_added_process, last_process_execution):
        if last_process_execution:
            origin = last_process_execution.destination
        else:
            main_part_type = value_added_process.get_main_entity_entity_type()

            if not self.agent.entity_provider:
                return None
            provider_resources = self.agent.entity_provider[main_part_type]
            if len(provider_resources) > 1:
                raise NotImplementedError()

            origin = provider_resources[0]

        return origin

    async def _organize_main_transport_process(self, destination, main_process_execution,
                                               support_entity_type, main_part_entity_type, start_time, end_time,
                                               planned_process_executions=[]):
        """Organize the main transport from the current location to the location of demand (value_added_process)"""
        await_callback, requests, accepted_time_period = \
            await self._organize_main_transport_process_request(
                destination=destination, main_process_execution=main_process_execution,
                main_part_entity_type=main_part_entity_type, support_entity_type=support_entity_type,
                start_time=start_time, end_time=end_time, planned_process_executions=planned_process_executions)

        # print(f"Transport interim {True if await_callback else False}")
        if await_callback:
            transport_successful, process_execution_transport_with_provider = \
                await self._organize_main_transport_process_response(await_callback, requests)
        else:
            await self._skip_negotiation_round(amount=1)
            process_execution_transport_with_provider = []
            transport_successful = True  # no transport needed

        return transport_successful, process_execution_transport_with_provider, accepted_time_period

    def get_destination(self, value_added_process_execution):
        """Determine the destination for the next value_added_process"""

        destination = value_added_process_execution.origin
        if isinstance(destination, NonStationaryResource):
            destination = value_added_process_execution.destination

        return destination

    def set_support_resource(self, value_added_process_execution, value_added_process, last_process_executions,
                             support_entity_type=None):
        value_added_process_preconditions = self.value_added_processes_preconditions[value_added_process]

        if last_process_executions and value_added_process_preconditions["long_time_support_needed"]:

            support_resource = value_added_process_execution.get_support_resource(support_entity_type)
            if support_resource is None:
                raise Exception(str(self.agent.current_order),
                                [f.name for f in self.agent.current_order.features_requested],
                                value_added_process_execution.get_name(), value_added_process_execution.event_type)

        else:
            print("Support Resource is None")
            support_resource = None

        self.support_resource = support_resource

    async def _organize_main_transport_process_request(self, destination, main_process_execution, support_entity_type,
                                                       main_part_entity_type, start_time, end_time=None,
                                                       planned_process_executions=[]):
        """Organize the main (resource or part) transport process_execution"""

        transport_transfer_processes, entities = \
            self._get_transport_processes(main_process_execution, support_entity_type, main_part_entity_type,
                                          destination)

        planned_processes = [planned_process_execution.process
                             for planned_process_execution in planned_process_executions]
        transport_transfer_processes = [transport_transfer_process
                                        for transport_transfer_process in transport_transfer_processes
                                        if transport_transfer_process["process"] not in planned_processes]

        # print(self.agent.name, self.support_resource.name,
        #       [transport_p["process"].name for transport_p in transport_transfer_processes])
        if not transport_transfer_processes:
            return None, [], None

        # print("Start ...")
        await_callback, requests, accepted_time_period = \
            await self._get_negotiation_objects_transport(transport_transfer_processes, entities, start_time, end_time)

        return await_callback, requests, accepted_time_period

    def _get_transport_processes(self, main_process_execution, support_entity_type, main_part_entity_type,
                                 destination):
        """Get transport processes for the main_part entity_type of the support entity_type"""

        if main_process_execution:
            # organize main_part transport

            if main_part_entity_type is not None:
                parts = main_process_execution.get_parts()
                if parts:

                    entities = list(set(parts))
                else:
                    entities = []

            elif support_entity_type is not None:
                resources = main_process_execution.get_resources()
                if resources:
                    entities = list(set(resources))
                else:
                    entities = []

            else:
                return [], []

        else:
            return [], []
            parts = []
            possible_origins = self.agent.entity_provider[support_entity_type]

            if 0 == len(possible_origins) or 1 < len(possible_origins):
                raise NotImplementedError

        origin = self.current_location

        if origin == destination:
            return [], []
        transport_transfer_processes = self._get_transport_process_chain(origin, destination, support_entity_type)

        return transport_transfer_processes, entities

    def _get_transport_process_chain(self, origin, destination, support_entity_type) -> list[dict]:
        """Determine a process chain for the transport processes"""
        transport_transfer_processes = \
            self.agent.routing_service.get_transit_processes(origin, destination, support_entity_type, transfers=True)

        # determine if the transfer is required
        transfers_needed = False
        if not isinstance(transport_transfer_processes[-1]["origin"], ConveyorBelt) and support_entity_type is None:
            transfers_needed = True

        if transfers_needed:  # ToDo
            # case if:
            # ... the main_resource is not a conveyor belt
            # ... no support resource needed for the VAP itself
            processes = [process["process"]
                         for process in transport_transfer_processes]
            loading_process_d: dict = self.agent.routing_service.get_transfer_process(
                origin=origin,  # entity_entity_type=part_entity_type,
                support_entity_type=support_entity_type,
                level_differences_allowed=True)
            if loading_process_d["process"] not in processes:
                transport_transfer_processes = [loading_process_d] + transport_transfer_processes

            unloading_process_d: dict = self.agent.routing_service.get_transfer_process(
                destination=destination,  # entity_entity_type=part_entity_type,
                support_entity_type=support_entity_type,
                level_differences_allowed=True)
            if unloading_process_d["process"] not in processes:
                transport_transfer_processes = transport_transfer_processes + [unloading_process_d]

        return transport_transfer_processes

    def determine_expected_transport_time(self):
        """
        Determine the expected time the transport resource should be blocked to avoid conflicts with other main_parts
        block the transport_resource in the meantime.
        :return the expected_transport_time
        """
        # iterate through the value_added_processes and determine the processes that need an agv

        # ToDo: memoization
        value_added_processes = self.agent.current_work_order.get_value_added_processes_requested_lst()
        if value_added_processes:
            return 1
        else:
            return 0
        # value_added_processes_with_agv = \
        #     [value_added_process
        #      for value_added_process in value_added_processes
        #      for root_node in value_added_process.transformation_controller.get_root_nodes()
        #      if root_node.transformation_type_support()]
        # # ToDo: review the link (successor) of the value_added_processes
        #
        # # determine the expected transport_time
        # expected_transport_time = sum([value_added_process.get_estimated_process_lead_time()
        #                                for value_added_process in value_added_processes_with_agv])
        #
        # # ToDo: transport_times - longest_ways?
        # expected_transport_time = 2 * expected_transport_time
        # # ToDo: waiting_times ??
        #
        # return expected_transport_time

    async def _get_negotiation_objects_transport(self, transport_transfer_processes, entities, start_time,
                                                 end_time=None):
        """Get the callback for the transport_process_executions request"""

        provider_negotiation_objects = {}
        negotiation_object_ids = []
        start_time_64 = np.datetime64(start_time.replace(microsecond=0))
        start_time_64 = np.maximum(self.current_time64, start_time_64)
        if end_time is None:
            accepted_time_period = np.array([start_time_64, np.datetime64])
        else:
            end_time64 = np.datetime64(end_time.replace(microsecond=0))
            accepted_time_period = np.array([start_time_64, end_time64])

        requests = []
        providers = []

        transport_process_executions = [self.get_transport_process_execution(entities, transport_transfer_process_dict)
                                        for transport_transfer_process_dict in transport_transfer_processes]
        if transport_process_executions:
            issue_id = ProcessCallForProposal.next_id

        transport_access_needed = False
        long_time_reservation = {self.support_resource: (self.determine_expected_transport_time(),
                                                         transport_access_needed)}

        process_group = False
        if len(transport_process_executions) > 1:
            expected_process_execution_time_sum = 0

            process_group = True

        for transport_process_execution in transport_process_executions:
            expected_process_execution_time = int(np.ceil(round(transport_process_execution.get_max_process_time(), 1)))

            end_time_64 = start_time_64 + np.timedelta64(expected_process_execution_time, "s")
            if not process_group:
                adapted_end_time64 = end_time_64 + np.timedelta64(240, "s")
                if end_time_64 < adapted_end_time64:
                    adapted_end_time64 = end_time_64
                accepted_time_periods = np.array([[start_time_64, adapted_end_time64]], dtype='datetime64[s]')

                # print(transport_process_execution.identification, transport_process_execution.process.name,
                #       accepted_time_periods)

                preference = self.agent.current_work_order_preference.get_process_execution_preference(
                    expected_time_slot_duration=expected_process_execution_time,
                    reference_objects=transport_process_execution, accepted_time_periods=accepted_time_periods)

                process_call_for_proposal = \
                    ProcessCallForProposal(reference_cfp=None, sender_name=self.agent.name,
                                           request_type=CallForProposal.RequestType.FLEXIBLE,
                                           client_object=self.agent.current_order, order=self.agent.current_order,
                                           fixed_origin=transport_process_execution.origin, issue_id=issue_id,
                                           process_execution=transport_process_execution,
                                           preference=preference, long_time_reservation=long_time_reservation)
                requests.append(transport_process_execution.process)

                providers_batch = self.agent.NegotiationBehaviour.convert_providers(
                    transport_process_execution.get_possible_main_resource_entity_types())
                for provider in providers_batch:
                    provider_negotiation_objects.setdefault(provider,
                                                            []).append(process_call_for_proposal)
                providers.extend(providers_batch)

                negotiation_object_ids.append(process_call_for_proposal.identification)
            else:
                expected_process_execution_time_sum += expected_process_execution_time

            start_time_64 = end_time_64

        if process_group:
            fixed_origin = transport_process_executions[0].origin

            end_time_64 += np.timedelta64(240, "s")
            if accepted_time_period[1] == accepted_time_period[1]:
                if end_time_64 > accepted_time_period[1]:
                    end_time_64 = accepted_time_period[1]
            start_time_64 = np.datetime64(start_time.replace(microsecond=0))
            start_time_64 = np.maximum(self.current_time64, start_time_64)
            accepted_time_periods = np.array([[start_time_64, end_time_64]], dtype='datetime64[s]')

            preference = self.agent.current_work_order_preference.get_process_execution_preference(
                expected_time_slot_duration=expected_process_execution_time_sum,
                reference_objects=transport_process_executions, accepted_time_periods=accepted_time_periods)
            process_group_call_for_proposal = \
                ProcessGroupCallForProposal(reference_cfp=None, sender_name=self.agent.name,
                                            request_type=CallForProposal.RequestType.FLEXIBLE,
                                            client_object=self.agent.current_order, order=self.agent.current_order,
                                            fixed_origin=fixed_origin, issue_id=issue_id,
                                            process_executions=transport_process_executions,
                                            preference=preference, long_time_reservation=long_time_reservation)

            requests = [process_execution.process
                        for process_execution in transport_process_executions]  # ToDo
            for transport_process_execution in transport_process_executions:
                main_resource_entity_types = transport_process_execution.get_possible_resource_entity_types()
                providers = \
                        [provider_agent_name
                         for provider_agent_name in
                         self.agent.NegotiationBehaviour.convert_providers(main_resource_entity_types)
                         if "_as" in provider_agent_name or "_w" in provider_agent_name]  # ToDo: too specific

                if providers:
                    provider = providers[0]
                    break

            provider_negotiation_objects[provider] = [process_group_call_for_proposal]
            providers = [provider]

            negotiation_object_ids.append(process_group_call_for_proposal.identification)

        # print("ST ..")
        round_, skipped_to_end = await self.agent.agents.ready_for_next_round(agent_name=self.agent.name)
        # print("St...")
        possible_proposal_providers = self.agent.NegotiationBehaviour.convert_providers(list(set(providers)))
        self.agent.agents.set_await_proposal(possible_proposal_providers=possible_proposal_providers, round_=round_)

        for provider, negotiation_objects in provider_negotiation_objects.items():
            await self.agent.NegotiationBehaviour.call_for_proposal(negotiation_objects, [provider])
        await_callback = asyncio.create_task(self.agent.NegotiationBehaviour.await_callback(negotiation_object_ids))
        accepted_time_period[1] = end_time_64

        return await_callback, requests, accepted_time_period

    def get_transport_process_execution(self, entities, transport_transfer_process_dict):
        process: Process = transport_transfer_process_dict["process"]
        parts_involved = [(entity,) for entity in entities
                          if process.check_possible_input_part(entity)]
        resources_used = []
        # if self.current_resource_binding:
        #     if process.check_ability_to_perform_process_as_resource(self.current_resource_binding):
        #         resources_used = [(self.current_resource_binding,)]

        origin = transport_transfer_process_dict["origin"]
        destination = transport_transfer_process_dict["destination"]

        transport_process_execution = \
            ProcessExecution(event_type=ProcessExecution.EventTypes.PLAN, executed_start_time=None,
                             executed_end_time=None, process=process,
                             parts_involved=parts_involved, resulting_quality=1,
                             resources_used=resources_used,  # (support_resource,)] if support_resource else [
                             main_resource=None, origin=origin, destination=destination,
                             order=self.agent.current_work_order.order,
                             source_application=self.agent.source_application)

        return transport_process_execution

    async def _organize_main_transport_process_response(self, await_callback, requests):
        """Response of the main transport organization"""

        process_executions, best_time_slot_value_proposals = \
            await self._get_process_executions(await_callback, requests)

        if best_time_slot_value_proposals:
            successful = True
            if not process_executions:
                raise Exception("The negotiation process failed")

            # ToDo: consider the provider if needed
            process_executions_with_provider = append_provider_to_process_executions(process_executions,
                                                                                     None)

        else:
            successful = False
            process_executions_with_provider = []

        return successful, process_executions_with_provider

    async def _handle_process_executions_batch(self, process_executions_with_provider, resource_binding,
                                               main_part_entity_type: EntityType | None,
                                               value_added_process: ValueAddedProcess | None = None,
                                               value_added_process_execution: ProcessExecution | None = None):
        """Used to handle the process_executions organized to release them/ forward them to the environment"""

        process_executions_with_provider = _get_sorted_process_executions(process_executions_with_provider,
                                                                          tuple_=True)

        if main_part_entity_type:
            self.block_periods_work_order(value_added_process_execution, main_part_entity_type,
                                          process_executions_with_provider)

        if value_added_process is not None:
            self.value_added_process_process_executions[value_added_process] = process_executions_with_provider

        await self._release_process_execution_batch(process_executions_with_provider=process_executions_with_provider,
                                                    resource_binding=resource_binding)

        if value_added_process is not None:
            print(get_debug_str(self.agent.name, self.__class__.__name__) +
                  f" VAP organization finished with "
                  f"{len(self.value_added_process_process_executions[value_added_process])} PE's",
                  [pe[0].get_process_name() for pe in self.value_added_process_process_executions[value_added_process]],
                  value_added_process.name,
                  self.agent.agents.get_current_round())
        # logger.debug(f"One VAP organization finished")

    def block_periods_work_order(self, value_added_process_execution, main_part_entity_type,
                                 process_executions_with_provider):
        """Block the periods in the work order based
        Only process_executions that contain the main part are considered"""

        if value_added_process_execution is not None:
            parts = value_added_process_execution.get_parts()
        else:
            parts = process_executions_with_provider[0][0].get_parts()

        main_parts = [part for part in parts if part.entity_type.check_entity_type_match(main_part_entity_type)]
        if not main_parts:
            main_part = None
            if self.current_resource_binding is None:
                raise Exception
            if self.current_resource_binding[0] in process_executions_with_provider[0][0].get_resources():
                main_resource = self.current_resource_binding[0]
            else:
                raise Exception
        else:
            main_part = main_parts[0]
            main_resource = None

        for process_execution_to_block, provider_agent_name in process_executions_with_provider:
            if (main_part in process_execution_to_block.get_parts() and main_part is not None) or \
                    (main_resource is not None and main_resource in process_execution_to_block.get_resources()):
                self.agent.current_work_order.block_period(
                    start_time=np.datetime64(process_execution_to_block.executed_start_time),
                    end_time=np.datetime64(process_execution_to_block.executed_end_time),
                    blocker_name=provider_agent_name,
                    issue_id=process_execution_to_block.identification,
                    process_execution_id=process_execution_to_block.identification,
                    work_order_id=self.agent.current_work_order.identification)

    async def _release_process_execution_batch(self, process_executions_with_provider, resource_binding):
        """Release of a complete process_execution_batch to execution"""

        self.agent.process_executions_queue += \
            [(self.agent.name, process_execution)
             for process_execution, provider in process_executions_with_provider]

        # print(self.agent.name, "Sequence:", [pe.process.name for pe, pro in process_executions_with_provider])

        ontology = "Release"

        await self._handle_process_execution_batch(
            process_executions_with_provider=process_executions_with_provider,
            ontology=ontology, resource_binding=resource_binding)

    async def _cancel_process_execution_batch(self, process_executions_with_provider):
        """Cancellation of a process_execution_batch that is not complete and therefore not to execute"""

        ontology = "Cancel"
        await self._handle_process_execution_batch(process_executions_with_provider, ontology)

    async def _handle_process_execution_batch(self, process_executions_with_provider, ontology, resource_binding=None):

        if resource_binding is not None:
            await self.bind_resource(resource_binding)

        if not process_executions_with_provider:
            return

        provider_process_executions = self.get_provider_process_executions(process_executions_with_provider,
                                                                           ontology)

        for provider, process_executions in provider_process_executions.items():
            # print(provider, process_executions)
            # print(provider, [pe["ProcessExecution"].process.name for pe in process_executions])
            process_executions_object = ListCO(process_executions)

            self.agent.agents.store_communication_object(process_executions_object)
            msg_content = process_executions_object.identification
            msg_process_execution_batch_sent = self.agent.create_message({"to": provider,
                                                                          "body": msg_content,
                                                                          "metadata": {"performative": "inform",
                                                                                       "ontology": ontology,
                                                                                       "language": "OWL-S"}})

            await self.send(msg_process_execution_batch_sent)

    async def bind_resource(self, resource_binding):
        """Bind a resource which means link the resource to the order
        because the resource is needed for more than one process"""

        provider = self.agent.address_book[resource_binding[0]]
        resource_binding_object = ObjectCO(resource_binding)
        self.agent.agents.store_communication_object(resource_binding_object)
        msg_content = resource_binding_object.identification
        msg_resource_binding_sent = self.agent.create_message({"to": provider,
                                                               "body": msg_content,
                                                               "metadata": {"performative": "inform",
                                                                            "ontology": "ResourceBinding",
                                                                            "language": "OWL-S"}})
        await self.send(msg_resource_binding_sent)

    def get_provider_process_executions(self, process_executions_with_provider, ontology):
        provider_process_executions = {}
        for process_execution, origin_provider in process_executions_with_provider:
            # hand the process_execution object to the provider would be the correct way
            if process_execution.main_resource not in self.agent.address_book:
                raise Exception(process_execution.main_resource.name)

            if ontology == "Cancel":
                for resource in process_execution.get_resources():
                    if not origin_provider:
                        provider = self.agent.address_book[resource]
                    else:
                        print("Origin provider:", resource.name, origin_provider)
                        provider = origin_provider

                    provider_process_executions.setdefault(provider,
                                                           []).append(process_execution)
            elif ontology == "Release":
                if not origin_provider:
                    provider = self.agent.address_book[process_execution.main_resource]
                else:
                    provider = origin_provider

                process_execution_d = \
                    {"ProcessExecution": process_execution,
                     "Deviation Tolerance Time Delta": self.agent.deviation_tolerance_time_delta,
                     "Notification Time Delta": self.agent.notification_time_delta}
                provider_process_executions.setdefault(provider,
                                                       []).append(process_execution_d)

        return provider_process_executions

    async def _send_proposals_respond(self, proposals_to_propose, proposals_to_refuse):
        """Send the "response" proposals to the provider agent_control"""

        for refused_proposal, provider in proposals_to_refuse:
            # print(refused_proposal.process_executions_component)
            await self.agent.NegotiationBehaviour.reject_proposal([refused_proposal])
        for proposal_to_propose, provider in proposals_to_propose:
            # print(proposal_to_propose.process_executions_component)
            await self.agent.NegotiationBehaviour.accept_proposal([proposal_to_propose])

    async def _request_availability(self, value_added_processes) -> Optional[Dict]:
        """
        The method is used to determine when the value_added_processes are available,
        respectively when the entities (resources, parts) that are required to execute are available.
        :param value_added_processes: a list of value_added_processes that could be executed next
        :return: a dictionary of processes and their next possible execution start time (the time stamp is not secure
        but an advisory)
        """
        # Maybe this method should be called in parallel to negotiation rounds
        # Therefore the costs would be to could not get the best result

        list_communication_object = \
            AvailabilityInformationCO(processes=value_added_processes,
                                      start_time_stamp=self.agent.change_handler.get_current_time(),
                                      round=self.current_round)
        msg_content = list_communication_object
        providers = [self.agent.address_book["information_service_agent"]]
        await self.agent.send_msg(behaviour=self, receiver_list=providers, msg_body=msg_content,
                                  message_metadata={"performative": "request",
                                                    "ontology": "AVAILABILITY",
                                                    "language": "OWL-S"})

        msg_received = await self.agent.receive_msg(self, timeout=10, metadata_conditions={"ontology": "AVAILABILITY"})

        if msg_received is None:
            return None

        msg_content, msg_sender, msg_ontology, msg_performative = msg_received
        earliest_possible_execution_times: Dict[Process, datetime] = msg_content

        return earliest_possible_execution_times

    def _determine_next_possible_execution_start_time(self, earliest_possible_execution_times):
        return min(list(earliest_possible_execution_times.values()))

    async def on_end(self):
        await super().on_end()
        # stop agent from behaviour
        await self.agent.shut_down()
        #print(self.digital_twin.get_number_of_orders_not_finished())
        print('finished on end')
