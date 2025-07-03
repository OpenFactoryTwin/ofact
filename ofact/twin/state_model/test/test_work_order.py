from random import choice
from typing import TYPE_CHECKING

import ofact.twin.state_model.processes as p
import ofact.twin.state_model.entities as e
from ofact.planning_services.model_generation.persistence import deserialize_state_model, get_state_model_file_path

if TYPE_CHECKING:
    from ofact.twin.state_model.sales import Feature


state_model_file_name = get_state_model_file_path(project_path=r"files/", state_model_file_name="mini.xlsx",
                                                   path_to_model="")
state_model_generation_settings = {"customer_generation_from_excel": True, "order_generation_from_excel": True,
                                   "customer_amount": 10, "order_amount": 10}
state_model = deserialize_state_model(source_file_path=state_model_file_name, persistence_format="xlsx",
                                      state_model_generation_settings=state_model_generation_settings)

orders = state_model.get_orders()
feature_process_mapper = state_model.get_feature_process_mapper()


def _create_work_order():
    sales_order = _get_random_sales_order()
    features_with_value_added_processes = sales_order.get_features_with_value_added_processes()
    value_added_processes_requested = (
        p.WorkOrder.convert_features_to_value_added_processes_requested(features_with_value_added_processes,
                                                                        feature_process_mapper))
    value_added_processes_completed: dict[Feature: dict[int: list[p.ValueAddedProcess]]] = {}
    work_order = p.WorkOrder(value_added_processes_completed=value_added_processes_completed,
                             value_added_processes_requested=value_added_processes_requested, order=sales_order)
    return work_order


def _get_random_sales_order():
    return choice(orders)


def test_create_bill_of_materials():
    work_order = _create_work_order()
    bom = work_order.create_bill_of_materials()

    assert isinstance(bom, dict)

    bill_of_materials = list(bom.keys())
    assert isinstance(bill_of_materials[0], e.EntityType)


def test_get_finished_part_entity_type():
    work_order = _create_work_order()
    finished_entity_type = work_order.get_finished_part_entity_type()

    assert isinstance(finished_entity_type, e.EntityType)
