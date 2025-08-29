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

Create the responses associated with the kpi tables
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from datetime import datetime

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.helpers import colored_print
from ofact.planning_services.scenario_analytics.interface.helpers import argument_preparation

if TYPE_CHECKING:
    from ofact.planning_services.scenario_analytics.scenario_handling.multi import MultiScenariosHandler
    from ofact.planning_services.scenario_analytics.scenario_handling.single import SingleScenarioHandler


# === filter response ==================================================================================================
# ==Response builder class
class FilterResponse:
    def __init__(self):
        # np.nan etc. values are not allowed!!!
        self.orders = []
        self.products = []
        self.processes = []
        self.resources = []

    def add_order(self, id_: str, reference_value: str):
        new = {"id": int(id_), "referenceValue": reference_value}
        self.orders.append(new)

    def add_product(self, id_: str, reference_value: str):
        new = {"id": int(id_), "referenceValue": reference_value}
        self.products.append(new)

    def add_process(self, id_: str, reference_value: str):
        new = {"id": int(id_), "referenceValue": reference_value}
        self.processes.append(new)

    def add_resource(self, id_: str, reference_value: str):
        new = {"id": int(id_), "referenceValue": reference_value}
        self.resources.append(new)

    def get_response_dict(self):
        return {"orders": self.orders,
                "products": self.products,
                "processes": self.processes,
                "resources": self.resources}

    def get_response_dict_ids(self):
        return {"orders": [order_dict["id"] for order_dict in self.orders],
                "products": [product_dict["id"] for product_dict in self.products],
                "processes": [process_dict["id"] for process_dict in self.processes],
                "resources": [resources_dict["id"] for resources_dict in self.resources]}


# ==build response
def build_filter_response(args, multi_scenarios_handler: Optional[MultiScenariosHandler],
                          scenario_handler=None) -> (dict, FilterResponse):
    """
    A function that returns a FilterResponse with content based on the digital Twin

    Parameters
    ----------
    args: arguments that were sent with the request could be passed here
    multi_scenarios_handler: handles all scenarios
    scenario_handler: handles one scenario
    """
    colored_print(f"[API] filter requested")
    response = FilterResponse()

    if scenario_handler is None:
        scenario_name = args["scenario"]
        scenario_available = multi_scenarios_handler.is_scenario_available(scenario_name)
    else:
        scenario_available = True
    if not scenario_available:
        return response

    start_time = args["dateStart"]
    end_time = args["dateEnd"]

    if scenario_handler is None:
        scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario_name)
    filter_options = scenario_handler.get_filter_options(start_time, end_time)

    for order_id, reference_value in filter_options["order"].items():
        response.add_order(id_=order_id, reference_value=reference_value)
    for product_id, reference_value in filter_options["product"].items():
        response.add_product(id_=product_id, reference_value=reference_value)
    for process_id, reference_value in filter_options["process"].items():
        response.add_process(id_=process_id, reference_value=reference_value)
    for resource_id, reference_value in filter_options["resource"].items():
        response.add_resource(id_=resource_id, reference_value=reference_value)

    return response


# === order_view response ==============================================================================================
# ==Response builder class
class OrdersResponse:
    def __init__(self):
        self.orders = []

    @argument_preparation
    def add_order(self, id_: str = "-", reference_value: str = "-", number_of_pieces_absolute: int = 0,
                  number_of_pieces_relative: int = 0, customer: str = "-", starting_date: int = 0,
                  release_date: int = 0, completion_date: int = 0, planned_completion_date: int = 0,
                  order_status: str = "-", priority: int = 1,
                  delivery_reliability: str = "-", delivery_delay: int = 0, total_lead_time: int = 0,
                  total_waiting_time: int = 0, current_stock: int = 0, quality: int = 0, performance: int = 0,
                  source: str = "-"):
        new = {"id": id_,
               "referenceValue": reference_value,
               "numberOfPiecesAbsolute": number_of_pieces_absolute,
               "numberOfPiecesRelative": number_of_pieces_relative,
               "customer": customer,
               "releaseDate": release_date,
               "startingDate": starting_date,
               "completionDate": completion_date,
               "plannedCompletionDate": planned_completion_date,
               "orderStatus": order_status,
               "priority": priority,
               "deliveryReliability": delivery_reliability,
               "deliveryDelay": delivery_delay,
               "totalLeadTime": total_lead_time,
               "totalWaitingTime": total_waiting_time,
               "currentStock": current_stock,
               "quality": quality,
               "performance": performance,
               "source": source}

        self.orders.append(new)

    def get_response_dict(self):
        return {"orders": self.orders}


# ==build response
# @profile
async def build_orders_response(start_time: datetime, end_time: datetime,
                                order_ids_list: list[int], product_ids_list: list[int],
                                process_ids_list: list[int], resource_ids_list: list[int],
                                units, scenario_handler: SingleScenarioHandler) -> (dict, OrdersResponse):

    if not order_ids_list:
        response = OrdersResponse()
        return response

    order_view = (
        await scenario_handler.get_order_view_awaited(order_ids_list, product_ids_list,
                                                      process_ids_list, resource_ids_list,
                                                      start_time, end_time))
    response = get_orders_response(order_view)

    return response


def get_orders_response(order_view):
    response = OrdersResponse()

    order_view.dropna(subset=['Number of Pieces Absolute'], inplace=True)

    for order_id, order_row in order_view.iterrows():
        try:
            reference_value = order_row["Order Identifier"]
        except KeyError as e:
            print(f"Problem with order {order_id}")
            continue

        try:
            response.add_order(id_=order_id,
                               reference_value=reference_value,
                               number_of_pieces_absolute=order_row["Number of Pieces Absolute"],
                               number_of_pieces_relative=order_row["Number of Pieces Relative"],
                               customer=order_row["Customer Name"],
                               release_date=order_row["Release Time [s]"],
                               starting_date=order_row["Start Time [s]"],
                               completion_date=order_row["Start Time [s]"],
                               planned_completion_date=order_row["Planned End Time [s]"],
                               order_status=order_row["Order Status"],
                               delivery_reliability=order_row["Delivery Reliability"],
                               delivery_delay=order_row["Delivery Delay"],
                               total_lead_time=order_row["total_lead_time_wt"],
                               total_waiting_time=order_row["total_waiting_time"],
                               quality=order_row["Resulting Quality"],
                               performance=order_row["performance"],
                               source=order_row["Source"])
        except:
            print(f"Datengrundlage für {reference_value} zu gering. "
                  f"Ggf. hilft die Erhöhung des Dateneinladezeitraums/Betrachtungszeitraums")

    return response


# ==Response builder class
class OrdersSummaryResponse:
    def __init__(self):
        self.orders_summary = {}

    @argument_preparation
    def add_orders_summary(self, number_of_pieces_absolute: int = 0, number_of_pieces_relative: int = 0,
                           customer: str = "-", release_date: int = 0, starting_date: int = 0, completion_date: int = 0,
                           planned_completion_date: int = 0,
                           order_status: str = "-", priority: int = 1, delivery_reliability: str = "-",
                           delivery_delay: int = 0, total_lead_time: int = 0, total_waiting_time: int = 0,
                           current_stock: int = 0, quality: int = 0, performance: int = 0, source: str = "-"):
        self.orders_summary = {"numberOfPiecesAbsolute": number_of_pieces_absolute,
                               "numberOfPiecesRelative": number_of_pieces_relative,
                               "customer": customer,
                               "releaseDate": release_date,
                               "startingDate": starting_date,
                               "completionDate": completion_date,
                               "plannedCompletionDate": planned_completion_date,
                               "orderStatus": order_status,
                               "priority": priority,
                               "deliveryReliability": delivery_reliability,
                               "deliveryDelay": delivery_delay,
                               "totalLeadTime": total_lead_time,
                               "totalWaitingTime": total_waiting_time,
                               "currentStock": current_stock,
                               "quality": quality,
                               "performance": performance,
                               "source": source}

    def get_response_dict(self):
        return self.orders_summary


# @profile
# ==build response
async def build_orders_summary_response(start_time: datetime, end_time: datetime,
                                        order_ids_list: list[int], product_ids_list: list[int],
                                        process_ids_list: list[int], resource_ids_list: list[int],
                                        units, scenario_handler: SingleScenarioHandler) -> (
        dict, OrdersSummaryResponse):
    response = OrdersSummaryResponse()
    if not order_ids_list:
        return response

    order_view = (
        await scenario_handler.get_order_view_awaited(order_ids_list, product_ids_list,
                                                      process_ids_list, resource_ids_list,
                                                      start_time, end_time, all=True))

    order_view.dropna(subset=['Number of Pieces Absolute'], inplace=True)
    if order_view.empty:
        response.add_orders_summary()
        return response
    order_view.loc[order_view["Start Time [s]"] == order_view["Start Time [s]"], "Start Time [s]"] -= 7200
    order_view.loc[order_view["End Time [s]"] == order_view["End Time [s]"], "End Time [s]"] -= 7200
    order_view.loc[order_view["Planned End Time [s]"] == order_view["Planned End Time [s]"], "Planned End Time [s]"] \
        -= 7200

    order_row = order_view.loc[0]
    try:
        response.add_orders_summary(number_of_pieces_absolute=order_row["Number of Pieces Absolute"],
                                    number_of_pieces_relative=order_row["Number of Pieces Relative"],
                                    customer=order_row["Customer Name"],
                                    release_date=order_row["Release Time [s]"],
                                    starting_date=order_row["Start Time [s]"],
                                    completion_date=order_row["Start Time [s]"],
                                    planned_completion_date=order_row["Planned End Time [s]"],
                                    # order_status=order_row["Order Status"],
                                    delivery_reliability=order_row["Delivery Reliability"],
                                    delivery_delay=order_row["Delivery Delay"],
                                    total_lead_time=order_row["total_lead_time_wt"],
                                    total_waiting_time=order_row["total_waiting_time"],
                                    quality=order_row["Resulting Quality"],
                                    performance=order_row["performance"],
                                    source=order_row["Source"])
    except:
        print("Problem with order summary")

    return response


# === product_view response ============================================================================================
# ==Response builder class
class ProductsResponse:
    def __init__(self):
        self.products = []

    @argument_preparation
    def add_product(self, id_: str = "-", reference_value: str = "-", target_quantity: int = 0,
                    quantity_produced: int = 0, difference_percentage: int = 0, product_shares: int = 0,
                    delivery_reliability: str = "-", lead_time: int = 0, total_lead_time: int = 0,
                    waiting_time: int = 0, total_waiting_time: int = 0, current_stock: int = 0, quality: int = 0,
                    performance: int = 0, source: str = "-"):
        new = {"id": id_,
               "referenceValue": reference_value,
               "targetQuantity": target_quantity,
               "quantityProduced": quantity_produced,
               "differencePercentage": difference_percentage,
               "productShares": product_shares,
               "deliveryReliability": delivery_reliability,
               "leadTime": lead_time,
               "totalLeadTime": total_lead_time,
               "waitingTime": waiting_time,
               "totalWaitingTime": total_waiting_time,
               "currentStock": current_stock,
               "quality": quality,
               "performance": performance,
               "source": source}
        self.products.append(new)

    def get_response_dict(self):
        return {"products": self.products}


# ==build response
# @profile
async def build_products_response(start_time: datetime, end_time: datetime,
                                  order_ids_list: list[int], product_ids_list: list[int],
                                  process_ids_list: list[int], resource_ids_list: list[int],
                                  units, scenario_handler: SingleScenarioHandler) -> (dict, ProductsResponse):
    product_view = (
        await scenario_handler.get_product_view_awaited(order_ids_list, product_ids_list,
                                                        process_ids_list, resource_ids_list,
                                                        start_time, end_time))

    response = get_products_response(product_view)

    return response


def get_products_response(product_view):
    response = ProductsResponse()
    product_view.dropna(subset=['Number of Pieces Absolute'], inplace=True)
    product_view = product_view.loc[product_view['Number of Pieces Absolute'] != 0]
    for product_id, product_row in product_view.iterrows():
        reference_value = product_row["Entity Type Name"]
        response.add_product(id_=product_id,
                             reference_value=reference_value,
                             target_quantity=product_row["Target Quantity"],
                             quantity_produced=product_row["Number of Pieces Absolute"],
                             difference_percentage=product_row["Difference Percentage"],
                             product_shares=product_row["Number of Pieces Relative"],
                             delivery_reliability=product_row["Delivery Reliability"],
                             lead_time=product_row["avg_lead_time_wt"],
                             waiting_time=product_row["avg_waiting_time"],
                             total_lead_time=product_row["total_lead_time_wt"],
                             total_waiting_time=product_row["total_waiting_time"],
                             quality=product_row['Resulting Quality'],
                             current_stock=product_row["Inventory"],
                             performance=product_row["performance"],
                             source=product_row["Source"])
    return response


# ==Response builder class
class ProductsSummaryResponse:
    def __init__(self):
        self.products_summary = {}

    @argument_preparation
    def add_product_summary(self, target_quantity: int = 0, quantity_produced: int = 0, difference_percentage: int = 0,
                            product_shares: int = 0, delivery_reliability: str = "-", lead_time: int = 0,
                            total_lead_time: int = 0, waiting_time: int = 0, total_waiting_time: int = 0,
                            current_stock: int = 0, quality: int = 0, performance: int = 0, source: str = "-"):
        self.products_summary = {"targetQuantity": target_quantity,
                                 "quantityProduced": quantity_produced,
                                 "differencePercentage": difference_percentage,
                                 "productShares": product_shares,
                                 "deliveryReliability": delivery_reliability,
                                 "leadTime": lead_time,
                                 "totalLeadTime": total_lead_time,
                                 "waitingTime": waiting_time,
                                 "totalWaitingTime": total_waiting_time,
                                 "currentStock": current_stock,
                                 "quality": quality,
                                 "performance": performance,
                                 "source": source}

    def get_response_dict(self):
        return self.products_summary


# ==build response
# @profile
async def build_products_summary_response(start_time: datetime, end_time: datetime,
                                          order_ids_list: list[int], product_ids_list: list[int],
                                          process_ids_list: list[int], resource_ids_list: list[int],
                                          units, scenario_handler: SingleScenarioHandler) -> (
        dict, ProductsSummaryResponse):
    response = ProductsSummaryResponse()

    product_view = (
        await scenario_handler.get_product_view_awaited(order_ids_list, product_ids_list,
                                                        process_ids_list, resource_ids_list,
                                                        start_time, end_time, all=True))

    if product_view.empty:
        response.add_product_summary()
        return response
    product_row = product_view.loc[0]
    response.add_product_summary(target_quantity=product_row["Target Quantity"],
                                 quantity_produced=product_row["Number of Pieces Absolute"],
                                 difference_percentage=product_row["Difference Percentage"],
                                 product_shares=product_row["Number of Pieces Relative"],
                                 delivery_reliability=product_row["Delivery Reliability"],
                                 lead_time=product_row["avg_lead_time_wt"],
                                 waiting_time=product_row["avg_waiting_time"],
                                 total_lead_time=product_row["total_lead_time_wt"],
                                 total_waiting_time=product_row["total_waiting_time"],
                                 quality=product_row['Resulting Quality'],
                                 current_stock=product_row["Inventory"],
                                 performance=product_row["performance"],
                                 source=product_row["Source"])

    return response


# === process_view response ============================================================================================
# ===Response builder class
class ProcessesResponse:
    def __init__(self):
        self.processes = []

    @argument_preparation
    def add_process(self, id_: str = "-", reference_value: str = "-", absolute_frequency: int = 0,
                    process_share: int = 0, delivery_reliability: str = "-", lead_time: int = 0,
                    waiting_time: int = 0, min_lead_time: int = 0, min_waiting_time: int = 0, max_lead_time: int = 0,
                    max_waiting_time: int = 0, var_lead_time: int = 0, var_waiting_time: int = 0, quality: int = 0,
                    performance: int = 0, source: str = "-"):
        new = {"id": id_,
               "referenceValue": reference_value,
               "absoluteFrequency": absolute_frequency,
               "processShare": process_share,
               "deliveryReliability": delivery_reliability,
               "leadTime": lead_time,
               "waitingTime": waiting_time,
               "minLeadTime": min_lead_time,
               "minWaitingTime": min_waiting_time,
               "maxLeadTime": max_lead_time,
               "maxWaitingTime": max_waiting_time,
               "varianceLeadTime": var_lead_time,
               "varianceWaitingTime": var_waiting_time,
               "quality": quality,
               "performance": performance,
               "source": source}
        self.processes.append(new)

    def get_response_dict(self):
        return {"processes": self.processes}


# ==build response
# @profile
async def build_processes_response(start_time: datetime, end_time: datetime,
                                   order_ids_list: list[int], product_ids_list: list[int],
                                   process_ids_list: list[int], resource_ids_list: list[int],
                                   units, scenario_handler: SingleScenarioHandler) -> (dict, ProcessesResponse):
    process_view = (
        await scenario_handler.get_process_view_awaited(order_ids_list, product_ids_list,
                                                        process_ids_list, resource_ids_list,
                                                        start_time, end_time))

    response = get_processes_response(process_view)

    return response


def get_processes_response(process_view):
    response = ProcessesResponse()
    process_view = process_view.loc[process_view["count"] != 0]
    for process_id, process_row in process_view.iterrows():
        reference_value = process_row["Process Name"]
        response.add_process(id_=process_id,
                             reference_value=reference_value,
                             absolute_frequency=process_row["count"],
                             process_share=process_row["relative"],
                             delivery_reliability=process_row["Delivery Reliability"],
                             lead_time=process_row["avg_lead_time_wt"],
                             waiting_time=process_row["avg_waiting_time"],
                             min_lead_time=process_row["min_lead_time_wt"],
                             min_waiting_time=process_row["min_waiting_time"],
                             max_lead_time=process_row["max_lead_time_wt"],
                             max_waiting_time=process_row["max_waiting_time"],
                             var_lead_time=process_row["var_lead_time_wt"],
                             var_waiting_time=process_row["var_waiting_time"],
                             quality=process_row["Resulting Quality"],
                             performance=process_row["performance"],
                             source=process_row["Source"])

    return response

# ===Response builder class
class ProcessesSummaryResponse:
    def __init__(self):
        self.processes_summary = {}

    @argument_preparation
    def add_process_summary(self, absolute_frequency: int = 0, process_share: int = 0,
                            delivery_reliability: str = "-", lead_time: int = 0, waiting_time: int = 0,
                            min_lead_time: int = 0, min_waiting_time: int = 0, max_lead_time: int = 0,
                            max_waiting_time: int = 0, var_lead_time: int = 0, var_waiting_time: int = 0,
                            quality: int = 0, performance: int = 0, source: str = "-"):
        self.processes_summary = {"absoluteFrequency": absolute_frequency,
                                  "processShare": process_share,
                                  "deliveryReliability": delivery_reliability,
                                  "leadTime": lead_time,
                                  "waitingTime": waiting_time,
                                  "minLeadTime": min_lead_time,
                                  "minWaitingTime": min_waiting_time,
                                  "maxLeadTime": max_lead_time,
                                  "maxWaitingTime": max_waiting_time,
                                  "varianceLeadTime": var_lead_time,
                                  "varianceWaitingTime": var_waiting_time,
                                  "quality": quality,
                                  "performance": performance,
                                  "source": source}

    def get_response_dict(self):
        return self.processes_summary


# ==build response
# @profile
async def build_processes_summary_response(start_time: datetime, end_time: datetime,
                                           order_ids_list: list[int], product_ids_list: list[int],
                                           process_ids_list: list[int], resource_ids_list: list[int],
                                           units, scenario_handler: SingleScenarioHandler) -> (
        dict, ProcessesSummaryResponse):
    process_view = (
        await scenario_handler.get_process_view_awaited(order_ids_list, product_ids_list,
                                                        process_ids_list, resource_ids_list,
                                                        start_time, end_time, all=True))

    response = ProcessesSummaryResponse()
    if not process_view["count"].sum():
        response.add_process_summary()
        return response

    process_row = process_view.loc[0]
    response.add_process_summary(absolute_frequency=process_row["count"],
                                 process_share=process_row["relative"],
                                 delivery_reliability=process_row["Delivery Reliability"],
                                 lead_time=process_row["avg_lead_time_wt"],
                                 waiting_time=process_row["avg_waiting_time"],
                                 min_lead_time=process_row["min_lead_time_wt"],
                                 min_waiting_time=process_row["min_waiting_time"],
                                 max_lead_time=process_row["max_lead_time_wt"],
                                 max_waiting_time=process_row["max_waiting_time"],
                                 var_lead_time=process_row["var_lead_time_wt"],
                                 var_waiting_time=process_row["var_waiting_time"],
                                 quality=process_row["Resulting Quality"],
                                 performance=process_row["performance"],
                                 source=process_row["Source"])

    return response


# === resource_view response ===========================================================================================
# ==Response builder class
class ResourcesResponse:
    def __init__(self):
        self.resources = []

    @argument_preparation
    def add_resource(self, id_: str = "-", reference_value: str = "-", process_frequency: int = 0,
                     resource_share: int = 0, delivery_reliability: str = "-", lead_time: int = 0,
                     waiting_time: int = 0, stock: int = 0, quality: int = 0, performance: int = 0,
                     total_resource_utilisation: int = 0, total_resource_availability: int = 0, ore: int = 0,
                     source: str = "-"):
        new = {"id": id_,
               "referenceValue": reference_value,
               "processFrequency": process_frequency,
               "resourceShare": resource_share,
               "deliveryReliability": delivery_reliability,
               "leadTime": lead_time,
               "waitingTime": waiting_time,
               "stock": stock,
               "quality": quality,
               "performance": performance,
               "totalResourceUtilisation": total_resource_utilisation,
               "totalResourceAvailability": total_resource_availability,
               "kpiOre": ore,
               "source": source}
        self.resources.append(new)

    def get_response_dict(self):
        return {"resources": self.resources}


# ==build response
# @profile
async def build_resources_response(start_time: datetime, end_time: datetime,
                                   order_ids_list: list[int], product_ids_list: list[int],
                                   process_ids_list: list[int], resource_ids_list: list[int],
                                   units, scenario_handler: SingleScenarioHandler) -> (dict, ResourcesResponse):

    resource_view_df = (
        await scenario_handler.get_resource_view_awaited(order_ids_list, product_ids_list,
                                                         process_ids_list, resource_ids_list,
                                                         start_time, end_time))

    response = get_resources_response(resource_view_df)

    return response


def get_resources_response(resource_view_df):
    response = ResourcesResponse()
    resource_view_df = resource_view_df.loc[resource_view_df["count"] != 0]
    resource_view_df.loc[resource_view_df["Inventory"] != resource_view_df["Inventory"], "Inventory"] = 0
    if resource_view_df.empty:
        response.add_resource()
        return response

    for resource_id, resource_row in resource_view_df.iterrows():
        reference_value = resource_row["Resource Used Name"]

        response.add_resource(id_=resource_id,
                              reference_value=reference_value,
                              process_frequency=resource_row["count"],
                              resource_share=resource_row["relative"],
                              delivery_reliability=resource_row["Delivery Reliability"],
                              lead_time=resource_row["avg_lead_time_wt"],
                              waiting_time=resource_row["avg_waiting_time"],
                              total_resource_utilisation=resource_row["utilisation"],
                              total_resource_availability=resource_row["Availability"],
                              quality=resource_row['Resulting Quality'],
                              performance=resource_row["performance"],
                              stock=resource_row["Inventory"],
                              ore=resource_row["ORE"],
                              source=resource_row["Source"])

    return response


# ==Response builder class
class ResourcesSummaryResponse:
    def __init__(self):
        self.resources_summary = {}

    @argument_preparation
    def add_resource_summary(self, process_frequency: int = 0, resource_share: int = 0,
                             delivery_reliability: str = "-", lead_time: int = 0, waiting_time: int = 0,
                             stock: int = 0, quality: int = 0, performance: int = 0,
                             total_resource_utilisation: int = 0, total_resource_availability: int = 0, ore: int = 0,
                             source: str = "-"):
        self.resources_summary = {"processFrequency": process_frequency,
                                  "resourceShare": resource_share,
                                  "deliveryReliability": delivery_reliability,
                                  "leadTime": lead_time,
                                  "waitingTime": waiting_time,
                                  "stock": stock,
                                  "quality": quality,
                                  "performance": performance,
                                  "totalResourceUtilisation": total_resource_utilisation,
                                  "totalResourceAvailability": total_resource_availability,
                                  "kpiOre": ore,
                                  "source": source}

    def get_response_dict(self):
        return self.resources_summary


# ==build response
# @profile
async def build_resources_summary_response(start_time: datetime, end_time: datetime,
                                           order_ids_list: list[int], product_ids_list: list[int],
                                           process_ids_list: list[int], resource_ids_list: list[int],
                                           units, scenario_handler: SingleScenarioHandler) -> (
        dict, ResourcesSummaryResponse):
    response = ResourcesSummaryResponse()
    resource_view_df = (
        await scenario_handler.get_resource_view_awaited(order_ids_list, product_ids_list,
                                                         process_ids_list, resource_ids_list,
                                                         start_time, end_time, all=True))

    if not resource_view_df["count"].sum():
        response.add_resource_summary()
        return response

    # if 0 not in kpi_inventory.index:
    #     kpi_inventory.loc[0] = 0

    resource_row = resource_view_df.loc[0]
    response.add_resource_summary(process_frequency=resource_row["count"],
                                  resource_share=resource_row["relative"],
                                  delivery_reliability=resource_row["Delivery Reliability"],
                                  lead_time=resource_row["avg_lead_time_wt"],
                                  waiting_time=resource_row["avg_waiting_time"],
                                  total_resource_utilisation=resource_row["utilisation"],
                                  total_resource_availability=resource_row["Availability"],
                                  quality=resource_row['Resulting Quality'],
                                  performance=resource_row["performance"],
                                  stock=resource_row["Inventory"],
                                  ore=resource_row["ORE"],
                                  source=resource_row["Source"])

    return response
