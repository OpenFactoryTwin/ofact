"""
Contains the order pool behaviour responsible for the order release
@last update: 21.08.2023
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import logging
from abc import abstractmethod, ABCMeta
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour
from ofact.twin.agent_control.helpers.communication_objects import ObjectCO
from ofact.twin.agent_control.helpers.debug_str import get_debug_str
from ofact.twin.state_model.model import get_orders_in_progress
from ofact.twin.state_model.sales import Order

# Module-Specific Constants
logger = logging.getLogger("OrderPool")


def _get_unfinished_orders(order_pool):
    # features_requested is not really usable for the project Schmaus

    unfinished_order_pool = [order
                             for order in order_pool
                             if order.delivery_date_actual is None]

    return unfinished_order_pool


def biased(input_):
    return input_ + 1e-12


class OrderPool(DigitalTwinCyclicBehaviour):
    """
    The order pool behaviour is used to release orders requested before the order agents.
    The release sequence is by default determined by the planned_delivery_date.
    It can be used to end the simulation if a target number of orders is reached.
    """

    def __init__(self):
        """
        :attribute orders_limit: The maximal number of orders that are released
        :attribute target_con_wip: The target number of orders that build the con_wip after the initial orders
        are completed.
        """
        super(OrderPool, self).__init__()

        request_template = {"metadata": {"performative": "request",
                                         "ontology": "ORDER",
                                         "language": "OWL-S"}}
        self.templates = [request_template]

        self.metadata_conditions = {"performative": "request", "ontology": "ORDER", "language": "OWL-S"}
        self.order_pool: Optional[list[Order]] = None

        self.orders_limit: Optional[int] = None  # limit of orders that define the simulation_end if set

        # object that track the progress and can send them to the front end
        self.number_of_orders_at_start = None  # if the value is static
        self.progress_tracker = None

        self.capacity_control_behaviour = NoCapacityControl()
        self.release_behaviour = UrgencyBasedRelease()
        self.sequence_creation_behaviour = DeliveryDateBasedSequenceCreation()
        self.round_orders_released = {}

    def get_current_time(self):
        return self.agent.change_handler.get_current_time()

    async def run(self):
        await super().run()

        msg_received = await self.agent.receive_msg(self, timeout=10,
                                                    metadata_conditions=self.metadata_conditions)
        if not msg_received:
            return

        msg_content, requester_agent_name, msg_ontology, msg_performative = msg_received

        new_release_allowed, agent_message = self.capacity_control_behaviour.new_release_allowed(requester_agent_name)
        if not new_release_allowed:
            print("No new release allowed")
            await self._send_new_order(agent_message, requester_agent_name)
            return

        if self.order_pool is None:
            self.number_of_orders_at_start, self.order_pool, orders_with_planned_release = (
                self.sequence_creation_behaviour.get_order_pool(self.agent.order_pool, self.get_current_time()))
        else:
            self.order_pool, orders_with_planned_release = (
                self.sequence_creation_behaviour.update_order_pool(self.order_pool, self.get_current_time()))
        if orders_with_planned_release is not None:
            for order in orders_with_planned_release:
                self._plan_order_release(order)

        new_order = True
        new_order_or_earliest_order_date = None
        if self.agent.agents.current_round_id not in self.round_orders_released:
            self.round_orders_released[self.agent.agents.current_round_id] = 1
        elif self.round_orders_released[self.agent.agents.current_round_id] > 1:
            new_order_or_earliest_order_date = self.get_current_time() + timedelta(seconds=1)
            new_order = False
        else:
           self.round_orders_released[self.agent.agents.current_round_id] += 1

        if new_order:
            # query if an order exists
            self.order_pool, new_order_or_earliest_order_date = (
                self.release_behaviour.get_new_order(requester_name=requester_agent_name, order_pool=self.order_pool,
                                                     current_time=self.get_current_time(), agent_name=self.agent.name,
                                                     agents_organization=self.agent.agents,
                                                     change_handler=self.agent.change_handler))

        if isinstance(new_order_or_earliest_order_date, Order):
            self._release_order(new_order_or_earliest_order_date)
        # else:
        #     print("No new order")

        await self._check_termination()
        self.track_progress()

        await self._send_new_order(new_order_or_earliest_order_date, requester_agent_name)

    def _plan_order_release(self, order: Order):
        self.agent.plan_order_release(order)

    def reminder_for_order_release(self, order: Order):
        pass  # Maybe already handled

    def _release_order(self, order: Order) -> None:
        """
        Release the order to be completed.

        Parameters
        ----------
        order: order to release
        """
        release_date_actual = self.get_current_time()

        if order.release_date_actual is not None:
            if order.release_date_actual < release_date_actual:
                release_date_actual = order.release_date_actual

        order.release(release_date_actual=release_date_actual)

    async def _send_new_order(self, new_order_or_agent_message, requester_agent_name):
        """Send an order to the requester_agent"""

        providers = [requester_agent_name]

        if new_order_or_agent_message is not None:
            msg_body = ObjectCO(new_order_or_agent_message)

        else:
            msg_body = new_order_or_agent_message

        await self.agent.send_msg(behaviour=self, receiver_list=providers, msg_body=msg_body,
                                  message_metadata={"performative": "inform",
                                                    "ontology": "ORDER",
                                                    "language": "OWL-S"})

    def set_orders_limit(self, orders_limit):
        self.orders_limit = orders_limit

    def set_order_agent_target_quantity(self, target_quantity):
        settings = {"target_con_wip": target_quantity}
        self.capacity_control_behaviour.set_capacity_control_settings(settings=settings)

    async def _check_termination(self):
        if self.orders_limit is None:
            return

        orders_released = self.number_of_orders_at_start - len(self.order_pool)
        if orders_released == self.orders_limit or orders_released % 5:
            print("Order termination check", orders_released, self.orders_limit)

        if self.number_of_orders_at_start - len(self.order_pool) >= self.orders_limit:
            await self.agent.agents.end_simulation()
            await self.agent.change_handler.end_simulation()

    #     def get_sorted_order_pool(self):
    #         """Sort the order_pool according delivery_date_planned"""
    #         # ToDo: further sorting influences - throughput time of the order
    #         sorting_feature_order_mapping = {datetime.strptime(order.delivery_date_planned, '%Y-%m-%d'): order
    #                                          for order in self.digital_twin.order_pool}
    #         sorted_sorting_feature = sorted(list(sorting_feature_order_mapping.keys()))
    #         sorted_order_pool = [sorting_feature_order_mapping[sorting_feature]
    #                              for sorting_feature in sorted_sorting_feature]
    #         return sorted_order_pool

    def set_progress_tracker(self, progress_tracker):
        self.progress_tracker = progress_tracker

    def track_progress(self):
        if self.progress_tracker is None:
            return

        progress_value = self.get_progress_value()
        self.progress_tracker.announce(progress_value)

    def get_progress_value(self):
        """Determine the current progress and return them
        The progress can be determined through different approaches
        If the simulation is stopped through a time progress, the time progress can be mapped
        If the simulation is stopped through an order_pool progress, the order_pool progress
        """

        if self.agent.agents.simulation_end[0].name == "ORDER":
            current_progress = len(self.order_pool)
            progress_value = current_progress / self.number_of_orders_at_start
        elif self.agent.agents.simulation_end[0].name == "TIME_LIMIT":
            simulation_end = self.agent.agents.simulation_end[1][1]
            simulation_start = self.agent.agents.simulation_end[1][0]
            whole_simulation_time_delta = simulation_end - simulation_start
            current_progress_time_delta = self.agent.agents.get_current_time() - simulation_start
            progress_value = 100 * round(biased(current_progress_time_delta.total_seconds()) /
                                   biased(whole_simulation_time_delta.total_seconds()), 1)
        else:
            raise NotImplemented("Choose another simulation and or implement another method.")

        progress_value *= 100  # percent
        return progress_value


class CapacityControl(metaclass=ABCMeta):

    def __init__(self):
        self.active_oder_agents = set()

    @abstractmethod
    def set_capacity_control_settings(self, settings):
        pass

    @abstractmethod
    def new_release_allowed(self, requester_agent_name):
        pass


class ConWIPCapacityControl(CapacityControl):

    def __init__(self):
        super().__init__()
        self.target_con_wip: Optional[int] = None

    def set_capacity_control_settings(self, settings):
        self.target_con_wip = settings["target_con_wip"]

    def new_release_allowed(self, requester_agent_name):
        new_release_allowed = self._check_con_wip_state(requester_agent_name)
        if new_release_allowed:
            msg_body = "REQUIRED"  # ToDo: datetime
        else:
            msg_body = None

        return new_release_allowed, msg_body

    def _check_con_wip_state(self, requester_agent_name) -> bool:

        if requester_agent_name not in self.active_oder_agents:
            self.active_oder_agents.add(requester_agent_name)
            return True

        elif requester_agent_name in self.active_oder_agents:
            if len(self.active_oder_agents) > self.target_con_wip:
                self.active_oder_agents.remove(requester_agent_name)
                return False

        return True


class NoCapacityControl(CapacityControl):

    def set_capacity_control_settings(self, settings):
        pass

    def new_release_allowed(self, requester_agent_name):
        new_release_allowed = True
        msg_body = None

        return new_release_allowed, msg_body


class Release(metaclass=ABCMeta):

    def __init__(self):
        self.last_request = {"round": 0,
                             "requesters": []}

    @abstractmethod
    def get_new_order(self, requester_name, order_pool, current_time, agent_name, agents_organization, change_handler):
        pass


class UrgencyBasedRelease(Release):

    def get_new_order(self, requester_name, order_pool, current_time, agent_name, agents_organization, change_handler):
        """
        Used to determine the next order.
        To achieve this, currently the order with the earliest delivery_date_planned is taken
        (but other selection methods are also possible).

        Returns
        -------
        next_order (Order | None): Next Order
        """

        next_order: Order | None = None
        order_not_ordered_until_now = []

        if order_pool:
            # choose the most urgent order
            while order_pool:
                next_order: Order = order_pool.pop(0)

                if next_order.release_date_planned is not None:
                    if next_order.release_date_planned > current_time:
                        # for orders that are not ordered at the time, the order agent request the order
                        order_not_ordered_until_now.append(next_order)
                        next_order: None = None
                        continue
                # consider the creation date of an order!
                elif next_order.order_date is not None:
                    if next_order.order_date > current_time:
                        # for orders that are not ordered at the time, the order agent request the order
                        order_not_ordered_until_now.append(next_order)
                        next_order: None = None
                        continue

                if next_order.features_requested:  # check the order_pool
                    break

            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       f" Orders to process: {len(self.order_pool)}")
        else:
            # case order_pool is empty
            print(get_debug_str(agent_name, self.__class__.__name__) + " Order Pool empty")

        if order_not_ordered_until_now:
            order_pool = order_not_ordered_until_now + order_pool

        # handle the case that no order is currently available but maybe in a few minutes
        # since the order date is in the future
        if next_order is None and order_pool:

            order_dates = [order.order_date
                           for order in order_pool
                           if order.order_date is not None]
            next_order = min(order_dates)
            # print("Determine earliest order creation date of the current order pool.",
            #       self.get_current_time(), next_order)

            # if all order agents has requested an order last round, change the current time ..
            if self.last_request["round"] < agents_organization.current_round_id:
                responder_requester_agent_names, responder_provider_agent_names = (
                    agents_organization.get_responder_agent_names())
                amount_requester_agents = len(responder_requester_agent_names)

                if len(self.last_request["requesters"]) == amount_requester_agents:
                    change_handler.set_current_time(next_order)

                # reset
                self.last_request = {"round": agents_organization.current_round_id,
                                     "requesters": []}

            self.last_request["requesters"].append(requester_name)

        return order_pool, next_order


class SequenceCreation(metaclass=ABCMeta):

    @abstractmethod
    def get_order_pool(self, order_pool=None, current_time=None):
        pass

    def update_order_pool(self, order_pool=None, current_time=None):
        return order_pool, None


class DeliveryDateBasedSequenceCreation(SequenceCreation):

    def __init__(self):
        pass

    def get_order_pool(self, order_pool=None, current_time=None):

        order_pool = order_pool.copy()

        order_pool = _get_unfinished_orders(order_pool)
        orders_with_planned_release = self._make_release_plan(order_pool, current_time)
        order_pool = list(set(order_pool))
        orders_already_started = get_orders_in_progress(order_pool, at=current_time)
        order_not_started = list(set(order_pool).difference(set(orders_already_started)))
        orders_already_started = self._sort_order_pool(orders_already_started)
        order_not_started = self._sort_order_pool(order_not_started)
        order_pool = orders_already_started + order_not_started

        number_of_orders_at_start = len(order_pool)
        return number_of_orders_at_start, order_pool, orders_with_planned_release

    def _make_release_plan(self, order_pool, current_time=None):
        order_for_release_planned = [order
                                     for order in order_pool
                                     if order.release_date_planned is not None and
                                     order.release_date_actual is None]
        orders_with_planned_release = [order
                                       for order in order_for_release_planned
                                       if order.release_date_planned > current_time]
        return orders_with_planned_release

    def _sort_order_pool(self, order_pool):
        # _according_delivery_date_planned
        orders_with_delivery_date_planned = [order
                                             for order in order_pool
                                             if order.delivery_date_planned is not None]
        orders_without_delivery_date_planned = [order
                                                for order in order_pool
                                                if order.delivery_date_planned is None]

        order_pool = sorted(orders_with_delivery_date_planned, key=lambda order: order.delivery_date_planned)
        order_pool += orders_without_delivery_date_planned
        orders_with_release_date = [order
                                    for order in order_pool
                                    if order.release_date_actual is not None]
        orders_without_release_date = [order
                                       for order in order_pool
                                       if order.release_date_actual is None]
        order_pool = orders_with_release_date + orders_without_release_date

        return order_pool
