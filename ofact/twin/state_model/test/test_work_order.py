from random import choice
from typing import TYPE_CHECKING

from ofact.planning_services.model_generation.twin_generator import get_digital_twin

import ofact.twin.state_model.processes as p
import ofact.twin.state_model.entities as e

if TYPE_CHECKING:
    from ofact.twin.state_model.sales import Feature


digital_twin_file_name = "mini.xlsx"
digital_twin_model = get_digital_twin(r"files/", digital_twin_file_name, path_to_model="", pickle_=False,
                                      customer_generation_from_excel=True, order_generation_from_excel=True,
                                      customer_amount=10, order_amount=10,)

orders = digital_twin_model.get_orders()
feature_process_mapper = digital_twin_model.get_feature_process_mapper()


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
