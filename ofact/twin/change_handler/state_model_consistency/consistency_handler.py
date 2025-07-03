"""
Check the consistency of the objects that should be stored in the state model subsequently.
"""

# Imports Part 1: Standard Imports
from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING

# Imports Part 3: Project Imports
from ofact.twin.state_model.processes import ProcessExecution
from ofact.twin.change_handler.state_model_consistency.validation.process_chain_validation import ProcessChainValidation
from ofact.twin.change_handler.state_model_consistency.validation.single_validator import validate_object

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.state_model.sales import Order

class SingleObjectConsistencyHandler:

    def ensure_process_execution_consistency(self, process_execution, completely_filled_enforced):
        completely_filled, not_completely_filled_attributes = validate_object(process_execution)

        if not completely_filled:
            debug_str = (f"[{self.__class__.__name__}] "
                         f"The execution of the process '{process_execution.get_name()}' with event type "
                         f"'{process_execution.event_type}' failed "
                         f"because the following attributes are missing: '{not_completely_filled_attributes}' \n"
                         f"{process_execution.__dict__}")

            # logging.debug(debug_str)
            if completely_filled_enforced:
                raise Exception(debug_str)

            else:
                print("Add PE PLAN (filled):", debug_str)

    def ensure_consistency(self, state_model_object):
        completely_filled, not_completely_filled_attributes = validate_object(state_model_object)

        # if not completely_filled:
        #     self._partially_filled_objects.add_partially_filled_object(
        #         object_=object_,
        #         not_completely_filled_attributes=not_completely_filled_attributes)


def _get_order_traces(process_executions):
    order_traces: dict[Order, list[ProcessExecution]] = {}
    for process_execution in process_executions:
        if process_execution.event_type != ProcessExecution.EventTypes.PLAN:
            continue

        order = process_execution.order
        if order in order_traces:
            order_traces[order].append(process_execution)
        else:
            order_traces[order] = [process_execution]

    return order_traces


class BatchConsistencyHandler:

    def __init__(self, state_model: StateModel):
        self.single_object_consistency_handler = SingleObjectConsistencyHandler()
        self.process_chain_validation = ProcessChainValidation(state_model=state_model)

    def validate_process_chain(self, process_executions: list[ProcessExecution]):
        order_traces = _get_order_traces(process_executions)
        self.process_chain_validation.validate_process_chain(order_traces)

    def validate_open_features(self, process_executions: list[ProcessExecution]):
        order_traces = _get_order_traces(process_executions)
        order_not_possible_features = self.process_chain_validation.validate_open_features(order_traces)
        return order_not_possible_features
