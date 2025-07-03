from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from enum import Enum

from ofact.twin.state_model.basic_elements import ProcessExecutionTypes

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel


class SmoothOperatorOptions(Enum):

    TRANSITION = ("Transition",
                  "RESOURCES in the transition are combined to a single resource")

    def __init__(self,
                 string: str,
                 description: str):
        self.string = string
        self.description = description


class SmoothOperator:

    def __init__(self, smooth_operator_option: SmoothOperatorOptions):
        self.smooth_operator_option = smooth_operator_option

    def smooth_state_model(self, state_model: StateModel):
        if self.smooth_operator_option == SmoothOperatorOptions.TRANSITION:
            state_model = self._smooth_resources(state_model)

        return state_model

    def _smooth_resources(self, state_model: StateModel):
        process_executions = state_model.get_process_executions_list(event_type=ProcessExecutionTypes.PLAN)
        print("PEs", len(process_executions))
        order_process_executions = {}
        for process_execution in process_executions:
            order_process_executions.setdefault(process_execution.order,
                                                []).append(process_execution)

        process_chain_elements = {}
        for order, process_executions in order_process_executions.items():
            if len(process_executions) <= 1:
                continue
            sorted_process_executions = sorted(process_executions,
                                               key=lambda pe: pe.executed_start_time
                                               if pe.executed_start_time else datetime(2100, 1, 1))

            process_chain = []
            for process_execution in sorted_process_executions:
                # same processes are assumed to be executed in parallel
                if process_execution.executed_start_time is None:
                    continue

                if process_chain:
                    if process_chain[-1]["process"] == process_execution.process:
                        if process_execution.origin not in process_chain[-1]["origins"]:
                            process_chain[-1]["origins"].append(process_execution.origin)
                        if process_execution.destination not in process_chain[-1]["destinations"]:
                            process_chain[-1]["destinations"].append(process_execution.destination)
                        continue
                process_chain.append({"process": process_execution.process,
                                      "origins": [process_execution.origin],
                                      "destinations": [process_execution.destination]})

            if len(process_chain) <= 1:
                continue

            for i in range(0, len(process_chain) - 1):
                first_process_element, second_process_element = process_chain[i:i + 2]

                element_combination = (first_process_element["process"],
                                       second_process_element["process"])
                if element_combination not in process_chain_elements:
                    process_chain_elements[element_combination] = {"first_destinations": [],
                                                                   "second_origins": []}

                process_chain_elements[element_combination]["first_destinations"].append(
                    first_process_element["destinations"])
                process_chain_elements[element_combination]["second_origins"].append(second_process_element["origins"])

        print([(process1.name, process2.name) for (process1, process2), _ in process_chain_elements.items()])

        return state_model
