from pathlib import Path
import pandas as pd
from ofact.twin.repository_services.deserialization.dynamic_state_model import DynamicStateModelDeserialization
from ofact.twin.state_model.basic_elements import ProcessExecutionTypes
from ofact.twin.state_model.seralized_model import SerializedStateModel
from projects.bicycle_world.settings import PROJECT_PATH

file_name = "six_orders.pkl"
state_model_sim_input_pkl_path = Path(str(PROJECT_PATH), f"scenarios/current/results/{file_name}")
state_model_dict = SerializedStateModel.load_from_pickle(state_model_sim_input_pkl_path)
importer = DynamicStateModelDeserialization()
state_model = importer.get_state_model(state_model_dict)

process_executions = state_model.get_process_executions_list(event_type=ProcessExecutionTypes.ACTUAL)


def _get_row(process_execution):
    row = {
        "Process Name": process_execution.process.name,
        "Start Time": process_execution.executed_start_time,
        "End Time": process_execution.executed_end_time,
        "Order Identifier": process_execution.order.identifier,
        "Main Resource": process_execution.main_resource.name,
        "Origin Resource": process_execution.origin.name if process_execution.origin else None,
        "Destination Resource": process_execution.destination.name if process_execution.destination else None,
        "Resulting Quality": process_execution.resulting_quality
    }
    row |= {f"Resource {i}": resource.name
            for i, resource in enumerate(process_execution.get_resources())}
    row |= {f"Part {i}": part.name
            for i, part in enumerate(process_execution.get_parts())}

    return row


execution_rows = [_get_row(process_execution)
                  for process_execution in process_executions]

execution_df = pd.DataFrame(execution_rows)

execution_df.to_excel("executions.xlsx")
