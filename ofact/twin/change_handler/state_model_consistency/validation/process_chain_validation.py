from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional

import pandas as pd

from ofact.planning_services.model_generation.persistence import deserialize_state_model

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.state_model.sales import Order, Feature
    from ofact.twin.state_model.processes import ProcessExecution, Process, ValueAddedProcess


def _convert_df(manual_mapping_df, processes):
    processes_by_name = {process.get_static_model_id()[1:]: process
                         for process in processes}

    manual_mapping_dict = {}
    for idx, row in manual_mapping_df.iterrows():
        try:
            processes_string_list = [processes_by_name[process_string]
                                     for process_string in eval(row["Possible Successors"])]
            manual_mapping_dict[processes_by_name[row["Process"]]] = processes_string_list
        except TypeError:  # not a relevant process
            pass
    return manual_mapping_dict


class ProcessChainValidation:

    def __init__(self, state_model: StateModel):
        self._state_model = state_model

        processes = self._state_model.get_all_processes()
        value_added_processes = self._state_model.get_value_added_processes()
        self.feature_process_mapper = self._state_model.get_feature_process_mapper()

        import os  # ToDo: should be project specific
        root_path = os.getcwd()
        if "ofact" in root_path:
            path_elements = root_path.split("ofact")
            ofact_count = root_path.count("ofact")
            root_path = path_elements[0]
            if ofact_count:  # the development code environment is named ofact
                root_path += "/ofact"
        elif "projects" in root_path:
            root_path = root_path.split("projects")[0]
        elif "/usr/src/app" in root_path:
            root_path = "/usr/src/app"
        else:
            raise Exception(f"Could not find a root path with ofact or projects in {root_path}")

        #if "Schmaus" not in str(os.getcwd()).rsplit("projects", 1)[1]:
        self.process_with_possible_process_successors: (
        dict[Union[Process, ValueAddedProcess], list[Union[Process, ValueAddedProcess]]]) = {}

        self.process_with_possible_value_added_process_successors: (
        dict[Union[Process, ValueAddedProcess], list[ValueAddedProcess]]) = {}
        self.validation_set_up_set = False
        return

       # else:
          #  self.validation_set_up_set = True

        path_to = "/projects/Schmaus/models/twin"
        chain_validation_path = f"{root_path.split('ofact')[0]}{path_to}/consistency_ensurance.xlsx"
        process_chain_manual_mapping_df: Optional[pd.DataFrame] = (
            pd.read_excel(chain_validation_path, sheet_name="ProcessChain"))
        vap_possibility_manual_mapping_df: Optional[pd.DataFrame] = (
            pd.read_excel(chain_validation_path, sheet_name="VAPPossibility"))

        if process_chain_manual_mapping_df is not None:
            process_with_possible_process_successors = _convert_df(process_chain_manual_mapping_df, processes)
        else:
            process_with_possible_process_successors = {
                process: self.get_possible_process_successors(process, processes)
                for process in processes}

        if vap_possibility_manual_mapping_df is not None:
            process_with_possible_value_added_process_successors = (
                _convert_df(vap_possibility_manual_mapping_df, processes))
        else:
            process_with_possible_value_added_process_successors = {
                process: self.get_possible_value_added_process_successors(process, value_added_processes)
                for process in processes}

        self.process_with_possible_process_successors: (
            dict[Union[Process, ValueAddedProcess], list[Union[Process, ValueAddedProcess]]]) = (
            process_with_possible_process_successors)

        self.process_with_possible_value_added_process_successors: (
            dict[Union[Process, ValueAddedProcess], list[ValueAddedProcess]]) = (
            process_with_possible_value_added_process_successors)

    def get_possible_process_successors(self, process: Union[Process, ValueAddedProcess],
                                        all_processes: list[Union[Process, ValueAddedProcess]]) -> (
            list[Union[Process, ValueAddedProcess]]):
        pass

    def get_possible_value_added_process_successors(self, process: Union[Process, ValueAddedProcess],
                                                    value_added_processes: list[ValueAddedProcess]) -> (
            list[Union[ValueAddedProcess]]):
        pass

    def validate_process_chain(self, process_chain: dict[Order, list[ProcessExecution]]):
        if not self.validation_set_up_set:
            return

        for order, process_executions_chain in process_chain.items():
            self._validate_process_executions_chain(process_executions_chain)

    def validate_open_features(self, process_chain: dict[Order, list[ProcessExecution]]) -> dict[Order, list[Feature]]:
        if not self.validation_set_up_set:
            return {order: []
                    for order, process_executions_chain in process_chain.items()}

        order_not_possible_features = {}
        for order, process_executions_chain in process_chain.items():
            not_possible_features = self._validate_open_value_added_processes(order, process_executions_chain)

            order_not_possible_features[order] = not_possible_features

        return order_not_possible_features

    def _validate_process_executions_chain(self, process_executions_chain: list[ProcessExecution]):
        """

        Parameters
        ----------
        process_executions_chain: list of process_execution assuming that they are sorted in time chronological order
        """

        processes = [process_execution.process
                     for process_execution in process_executions_chain]

        chain_error_occurred = False
        for i in range(len(processes) - 1):
            if (processes[i + 1] not in
                    self.process_with_possible_process_successors[processes[i]]):
                chain_error_occurred = True
                order = process_executions_chain[i].order
                print(f"Process '{processes[i + 1].name}' of order {order.identifier, order.external_identifications} "
                      f"is not a successor of {processes[i].name} (not handled)")

        if chain_error_occurred:
            process_names = [process_execution.get_name()
                             for process_execution in process_executions_chain]
            print(f"Process Executions involved: {process_names}")

    def _validate_open_value_added_processes(self, order, process_executions_chain: list[ProcessExecution]) -> (
            list[Feature]):
        """

        Parameters
        ----------
        process_executions_chain: list of process_execution assuming that they are sorted in time chronological order
        """
        not_possible_features = []

        if process_executions_chain:
            last_process_execution = process_executions_chain[-1]
            last_process = last_process_execution.process
        else:
            return not_possible_features

        features_requested: list[Feature] = order.features_requested

        value_added_processes_requested = [process
                                           for feature in features_requested
                                           for process in self.feature_process_mapper[feature]]
        if not self.validation_set_up_set:
            return not_possible_features

        value_added_processes_possible = self.process_with_possible_value_added_process_successors[last_process]

        inconsistent = not set(value_added_processes_requested).issubset(set(value_added_processes_possible))

        if not inconsistent:
            return not_possible_features

        for feature in features_requested:
            feature_inconsistent = \
                not set(self.feature_process_mapper[feature]).issubset(set(value_added_processes_possible))

            if feature_inconsistent:  # ToDo: checking and rechecking - problem too many orders are filtered ...
                print(f"The feature {feature.name} is not possible for order {order.identifier} ({order.external_identifications})")
                # not_possible_features.append(feature)

        return not_possible_features


if "__main__" == __name__:
    from ofact.twin.repository_services.persistence import deserialize_state_model

    state_model_path = "C:/Users/afreiter/PycharmProjects/digitaltwin/projects/Schmaus/data/raw_dt/test.pkl"
    state_model = deserialize_state_model(state_model_path)
    ProcessChainValidation(state_model=state_model)
