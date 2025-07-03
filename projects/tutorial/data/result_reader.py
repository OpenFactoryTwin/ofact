from __future__ import annotations

from pathlib import Path
import pandas as pd
from ofact.twin.repository_services.deserialization.dynamic_state_model import DynamicStateModelDeserialization
from ofact.twin.state_model.basic_elements import ProcessExecutionTypes
from ofact.twin.state_model.seralized_model import SerializedStateModel
from projects.bicycle_world.settings import PROJECT_PATH

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ofact.twin.state_model.sales import Order

file_name = "schedule_s1.pkl"
state_model_sim_input_pkl_path = Path(str(PROJECT_PATH), f"scenarios/current/results/{file_name}")
state_model_dict = SerializedStateModel.load_from_pickle(state_model_sim_input_pkl_path)
importer = DynamicStateModelDeserialization()
state_model = importer.get_state_model(state_model_dict)

process_executions = state_model.get_process_executions_list(event_type=ProcessExecutionTypes.PLAN)

def _handle_created_part(part_name):
    part_name = part_name.replace(" ", "_")
    part_name = part_name.replace(".", "")
    return "_" + part_name + "_pa"


def _get_row(process_execution):
    row = {
        "Process Execution ID": process_execution.identification,
        "Process Name": process_execution.process.get_static_model_id()[1:],
        "Start Time": process_execution.executed_start_time,
        "End Time": process_execution.executed_end_time,
        "Order Identifier": process_execution.order.identification,
        "Feature": process_execution.process.feature.get_static_model_id()[1:] if hasattr(process_execution.process, "feature") else None,
        "Main Resource": process_execution.main_resource.get_static_model_id()[1:],
        "Origin Resource": (process_execution.origin.get_static_model_id()[1:]
                            if process_execution.origin else None),
        "Destination Resource": (process_execution.destination.get_static_model_id()[1:]
                                 if process_execution.destination else None),
        "Resulting Quality": process_execution.resulting_quality
    }

    resources = process_execution.get_resources()[1:]
    row |= {f"Resource {i}": resource.get_static_model_id()[1:]
            for i, resource in enumerate(resources)}
    parts = process_execution.get_parts()
    row |= {f"Part {i}": (part.get_static_model_id()[1:]
                          if "static_model" in part.external_identifications
                          else _handle_created_part(part.name))
            # parts are created in the simulation (no material supply)
            for i, part in enumerate(parts)}
    # row |= {f"Part Type {i}": part.entity_type.get_static_model_id()
    #         for i, part in enumerate(parts)}

    return row


execution_rows = [_get_row(process_execution)
                  for process_execution in process_executions]

execution_df = pd.DataFrame(execution_rows)

execution_df.to_excel("executions.xlsx", index=False)

orders = state_model.get_orders()

def _get_order_row(order: Order):
    row = {
        "Identifier": order.identification,
        "Product Class": order.product_classes[0].get_static_model_id()[1:],
        "Price": order.price,
        "Customer": order.customer.get_static_model_id()[1:],
        "Order Date": order.order_date,
        "Release Date": order.release_date_actual,
        "Delivery Date Requested": order.delivery_date_requested,
        "Delivery Date Planned": order.delivery_date_planned,
        "Delivery Date Actual": order.delivery_date_actual,
        "Urgent": order.urgent
    }
    features = order.features_requested + order.features_completed
    row |= {f"Feature {i}": feature.get_static_model_id()[1:]
            for i, feature in enumerate(features)}
    return row


order_rows = [_get_order_row(order)
              for order in orders]

orders_df = pd.DataFrame(order_rows)

orders_df.to_excel("orders.xlsx", index=False)
