"""
#############################################################
This program and the accompanying materials are made available under the
terms of the Apache License, Version 2.0 which is available at
https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

SPDX-License-Identifier: Apache-2.0
#############################################################
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ofact.twin.state_model.sales import Order, Customer
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.state_model.processes import ProcessExecution


def _match_sales_objects_to_match_identification(sales_objects) -> dict[object:list[Order]]:
    match_identification_sales_objects = {}
    for sales_object in sales_objects:
        if not len(sales_object.external_identifications):  # assumption: each order has one external identification
            continue
        match_identification = list(sales_object.external_identifications.values())[0][0]
        match_identification_sales_objects_lst = \
            match_identification_sales_objects.setdefault(match_identification, [])
        match_identification_sales_objects_lst.append(sales_object)

    return match_identification_sales_objects


def _match_process_executions_to_match_identification(process_executions) -> dict[object:list[ProcessExecution]]:
    match_identification_process_executions = {}
    for process_execution in process_executions:
        if "match" not in process_execution.external_identifications:
            continue
        match_identification = process_execution.external_identifications["match"][0]
        match_identification_process_executions_lst = \
            match_identification_process_executions.setdefault(match_identification, [])
        match_identification_process_executions_lst.append(process_execution)

    return match_identification_process_executions


def _match_other_with_reference_sales_object(other_match_id_sales_objects,
                                             reference_match_id_sales_objects):
    other_reference_match = {}
    for other_match_id, sales_objects in other_match_id_sales_objects.items():
        if other_match_id not in reference_match_id_sales_objects:
            continue

        reference_sales_objects = reference_match_id_sales_objects[other_match_id]

        for idx, other_sales_object in enumerate(sales_objects):

            if not (len(reference_sales_objects) > idx):
                continue
            other_reference_match[other_sales_object] = reference_sales_objects[idx]

    return other_reference_match


def _match_other_with_reference_process_execution(other_match_id_process_executions,
                                                  reference_match_id_process_executions):
    other_reference_match = {}
    for other_match_id, process_executions in other_match_id_process_executions.items():
        if other_match_id not in reference_match_id_process_executions:
            continue

        reference_process_executions = reference_match_id_process_executions[other_match_id]

        for idx, other_process_execution in enumerate(process_executions):

            if not (len(reference_process_executions) > idx):
                continue
            other_reference_match[other_process_execution] = reference_process_executions[idx]

            if not other_process_execution.connected_process_execution:
                continue
            other_reference_match[other_process_execution.connected_process_execution] = \
                reference_process_executions[idx].connected_process_execution

            # if {Order ID}: {reference of the used process} blablabla
            # Pre-picking -> feature???
            # Position itself - how???
            # Transport processes {Order ID}: {reference of the used process}
            # Couple
            # Uncouple (: is missing)

    return other_reference_match


def _match_parts(other_parts_involved, reference_parts_involved):
    other_id_reference_id_match = {}
    etn_id_match_other, etn_id_match_none_other = _match_etn_with_parts(other_parts_involved)
    etn_id_match_reference, etn_id_match_none_reference = _match_etn_with_parts(reference_parts_involved)

    # in the first place try to match via etn and in the second with all others
    not_matched_others = []
    parts_reference_used = []
    for etn_id, part_other in etn_id_match_other.items():
        if etn_id not in etn_id_match_reference:
            not_matched_others.append(part_other)
            continue

        part_reference = etn_id_match_reference[etn_id]
        other_id_reference_id_match[part_other.identification] = part_reference.identification
        parts_reference_used.append(part_reference)

    not_matched_reference = [part_reference for etn_id, part_reference in etn_id_match_reference.items()
                             if part_reference not in parts_reference_used]
    not_matched_reference += etn_id_match_none_reference
    not_matched_others += etn_id_match_none_other

    for part_other in not_matched_others:
        for part_reference in not_matched_reference:
            if part_other.entity_type.identification == part_reference.identification:
                other_id_reference_id_match[part_other.identification] = part_reference.identification
                break

    return other_id_reference_id_match


def _match_etn_with_parts(parts_involved) -> []:
    etn_id_match_other = {}
    etn_id_match_other_none = []
    for part_tuple in parts_involved:
        if len(part_tuple) == 2:  # assumption same static model
            etn_id_match_other[part_tuple[1].identification] = part_tuple[0]
        else:
            etn_id_match_other_none.append(part_tuple[0])

    return etn_id_match_other, etn_id_match_other_none


class StateModelMatcher:

    def __init__(self, reference_state_model: StateModel, other_state_model: StateModel):
        self.reference_digital_twin = reference_state_model
        self.other_digital_twin = other_state_model

        self.orders_match = {}
        self.process_executions_match = {}

    def match_customers(self):
        reference_customers: list[Customer] = self.reference_digital_twin.customer_base
        other_customers: list[Customer] = self.other_digital_twin.customer_base

        reference_match_id_customers: dict[object:list[Order]] = \
            _match_sales_objects_to_match_identification(reference_customers)
        other_match_id_customers: dict[object:list[Order]] = \
            _match_sales_objects_to_match_identification(other_customers)

        other_reference_match = _match_other_with_reference_sales_object(reference_match_id_customers,
                                                                         other_match_id_customers)

        other_id_reference_id_match = {other.identification: reference.identification
                                       for other, reference in other_reference_match.items()}
        matchable_customers_ids = list(other_id_reference_id_match.keys())

        return other_id_reference_id_match, matchable_customers_ids

    def match_orders(self):
        reference_orders: list[Order] = self.reference_digital_twin.get_orders()
        other_orders: list[Order] = self.other_digital_twin.get_orders()

        reference_match_id_orders: dict[object:list[Order]] = \
            _match_sales_objects_to_match_identification(reference_orders)
        other_match_id_orders: dict[object:list[Order]] = \
            _match_sales_objects_to_match_identification(other_orders)

        other_reference_match = _match_other_with_reference_sales_object(reference_match_id_orders,
                                                                         other_match_id_orders)

        self.orders_match = other_reference_match
        other_id_reference_id_match = {other.identification: reference.identification
                                       for other, reference in other_reference_match.items()}

        matchable_order_ids = list(other_id_reference_id_match.keys())

        return other_id_reference_id_match, matchable_order_ids

    def match_products(self):
        """based on order_matches"""

        if not self.orders_match:
            self.match_orders()

        other_id_reference_id_match = {}
        for other_order, reference_order in self.orders_match.items():
            other_products = other_order.products
            reference_products = reference_order.products

            if not (other_products and reference_products):
                continue

            for other_product in other_products:
                for reference_product in reference_products:
                    if (other_product.entity_type.check_entity_type_match(reference_product.entity_type) and
                            other_product.identification not in other_id_reference_id_match):
                        other_id_reference_id_match[other_product.identification] = reference_product.identification

        matchable_product_ids = list(other_id_reference_id_match.keys())

        return other_id_reference_id_match, matchable_product_ids

    def match_process_executions(self) -> dict[int: int]:
        # current_state: matching_id in external identifications
        reference_process_executions: list[ProcessExecution] = self.reference_digital_twin.get_process_executions_list()
        other_process_executions: list[ProcessExecution] = self.other_digital_twin.get_process_executions_list()

        reference_match_id_process_executions: dict[object:list[ProcessExecution]] = \
            _match_process_executions_to_match_identification(reference_process_executions)
        other_match_id_process_executions: dict[object:list[ProcessExecution]] = \
            _match_process_executions_to_match_identification(other_process_executions)

        other_reference_match = _match_other_with_reference_process_execution(other_match_id_process_executions,
                                                                              reference_match_id_process_executions)

        self.process_executions_match = other_reference_match

        other_id_reference_id_match = \
            {other_process_execution.identification: reference_process_execution.identification
             for other_process_execution, reference_process_execution in other_reference_match.items()}

        matchable_process_execution_ids = list(other_id_reference_id_match.keys())

        return other_id_reference_id_match, matchable_process_execution_ids

    def match_parts(self):
        """Idea: matching through the process_executions"""

        if not self.process_executions_match:
            self.match_process_executions()

        other_id_reference_id_match = {}
        for other_process_execution, reference_process_execution in self.process_executions_match.items():
            other_parts_involved = other_process_execution.parts_involved
            reference_parts_involved = reference_process_execution.parts_involved

            other_id_reference_id_match_batch = _match_parts(other_parts_involved, reference_parts_involved)
            other_id_reference_id_match |= other_id_reference_id_match_batch

        matchable_part_ids = list(other_id_reference_id_match.keys())

        return other_id_reference_id_match, matchable_part_ids

    def get_first_unassigned_identification(self):
        first_unassigned_identification = self.reference_digital_twin.get_first_unassigned_identification()
        return first_unassigned_identification