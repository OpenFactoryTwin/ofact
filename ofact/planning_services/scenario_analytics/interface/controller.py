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

TODO response-object has default value in for ID - should this better be removed?
This Module contains all the API-specific content.
A class for each response-type is defined (necessary to build correct response objects for the frontend-api calls).
A Flask-Restful Server is created which is configured to answer requests
 on the Endpoint ServerIP:5000/api/v1/[filter,orders,products,processes,resources].
Further it will validate the requests (check for for missing parameters and their Type, but the parameter-content
 is not checked.
@author:Roman Sliwinski
@version:2021.10.25

----- Known issues -----
- Processing of requests might be time consuming. if the user does not wait and opens another page / changes the filter
 in between the first request and the response execution of processing should be stopped / no valid result should be sent
 1st) this costs resource-usage on the server
 2ed) this can bug the frontend
    e.g. filtersettingA -> refresh -> filtersettingB -> refresh -> resultB [time] -> resultA
    --> user will see result B for a moment, but it is replaced bei resultA witch does not match the users filtersetting
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import asyncio
import queue
import time
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
import flask.scaffold
from flask_cors import CORS

from ofact.planning_services.scenario_analytics.interface.services.scenario_export import (
    build_scenario_overview_response, build_scenario_export_response)
from ofact.planning_services.scenario_analytics.scenario_handling.single import SingleScenarioHandler
from ofact.planning_services.scenario_analytics.scenario_handling.multi import MultiScenariosHandler

flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from flask import Flask, request, Response, send_file
from flask_restful import Resource, Api, reqparse

# Imports Part 3: Project Imports
from ofact.planning_services.scenario_analytics.interface.settings.functions_used import DashboardControllerSettings
from ofact.helpers import colored_print
from ofact.planning_services.scenario_analytics.interface.services.enrichment import (
    build_digital_twin_current_state_available, build_digital_twin_enrichment_response)
from ofact.planning_services.scenario_analytics.interface.helpers import validate_request, get_datetime
from ofact.planning_services.scenario_analytics.interface.services.kpi_charts import (
    build_resources_utilization_chart_response, build_lead_time_chart_response)
from ofact.planning_services.scenario_analytics.interface.services.kpi_tables import (
    build_orders_response, build_filter_response, build_orders_summary_response, build_products_response,
    build_products_summary_response, build_processes_response, build_processes_summary_response,
    build_resources_response, build_resources_summary_response)
from ofact.planning_services.scenario_analytics.interface.services.simulation import (
    build_simulation_model_paths_response, build_simulation_response)

if TYPE_CHECKING:
    pass

# Module-Specific Constants

# initialize the kpi's
API_GET_ORDERS_SCHEME = DashboardControllerSettings.get_API_GET_ORDERS_SCHEME()
API_GET_PRODUCTS_SCHEME = DashboardControllerSettings.get_API_GET_PRODUCTS_SCHEME()
API_GET_PROCESSES_SCHEME = DashboardControllerSettings.get_API_GET_PROCESSES_SCHEME()
API_GET_RESOURCES_SCHEME = DashboardControllerSettings.get_API_GET_RESOURCES_SCHEME()
API_GET_UTILIZATION_CHART_RESOURCES_SCHEME = \
    DashboardControllerSettings.get_API_GET_UTILIZATION_CHART_RESOURCES_SCHEME()
API_GET_LEAD_TIME_CHART = DashboardControllerSettings.get_API_GET_LEAD_TIME_CHART()

API_GET_CURRENT_STATE_AVAILABLE = DashboardControllerSettings.get_API_GET_CURRENT_STATE_AVAILABLE()
API_SET_CURRENT_STATE_AVAILABLE = DashboardControllerSettings.get_API_SET_CURRENT_STATE_AVAILABLE()
API_GET_ENRICHED_TWIN_MODEL_SCHEME = DashboardControllerSettings.get_API_GET_ENRICHED_TWIN_MODEL_SCHEME()
API_GET_SIMULATION_PARAMETER_SCENARIO_SCHEME = (
    DashboardControllerSettings.get_API_GET_SIMULATION_PARAMETER_SCENARIO_SCHEME())
API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME = (
    DashboardControllerSettings.
    get_API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME())
API_GET_SIMULATION_SCHEME = DashboardControllerSettings.get_API_GET_SIMULATION_SCHEME()
API_HOST = DashboardControllerSettings.get_API_HOST()
API_PORT = DashboardControllerSettings.get_API_PORT()
PROJECT_PATH = DashboardControllerSettings.get_PROJECT_PATH()
DATA_SOURCE_MODEL_NAME = DashboardControllerSettings.get_DATA_SOURCE_MODEL_NAME()
DIGITAL_TWIN_FILE_PATH = DashboardControllerSettings.get_DIGITAL_TWIN_FILE_PATH()
get_digital_twin_updated_func = DashboardControllerSettings.get_update_digital_twin_func()
simulation_func = DashboardControllerSettings.get_simulation_func()
API_SCENARIO_EXPORT_PATH_SCHEME = ""
API_POST_SCENARIO_EXPORT_SCHEME = DashboardControllerSettings.get_API_POST_SCENARIO_EXPORT_SCHEME()


scenario_handler_current_state = SingleScenarioHandler()
multi_scenarios_handler: MultiScenariosHandler = (
    MultiScenariosHandler(scenarios_dict={"current_state": scenario_handler_current_state}))

scenario_model_paths = {}

app = Flask(__name__)
api = Api(app)

CORS(app)


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = Response()
        res.headers['X-Content-Type-Options'] = '*'
        return res


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Expose-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')  # GET,POST
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


# ====/api/v1/status====================================================================================================
# ==derive responses
class Status(Resource):
    def get(self):
        colored_print(f"[API] Got Get-Request on /status")
        # args = request.form
        response = {"status": "API Service v1.0 is working"}
        colored_print(f"[API]\tFilters response: {str(response)}")
        return response


# ==register in API
api.add_resource(Status, "/api/v1/status")

# ====/api/v1/filters====================================================================================================
# ==Request Parsers
filter_get_args = reqparse.RequestParser()
filter_get_args.add_argument("dateStart",
                             type=int,
                             help="unix-timestamp of begin of time window is required",
                             required=True,
                             location='args')
filter_get_args.add_argument("dateEnd",
                             type=int,
                             help="unix-timestamp of end of time window is required",
                             required=True,
                             location='args')
filter_get_args.add_argument("scenario",
                             type=str,
                             help="string",
                             required=True,
                             location='args')


# ==derive responses
class Filter(Resource):
    def get(self):
        args = filter_get_args.parse_args()
        colored_print(f"[API] Got Get-Request on /filters with arguments {str(args)}")
        response = build_filter_response(args, multi_scenarios_handler)
        response_dict = response.get_response_dict()
        colored_print(f"[API]\tFilters response: {str(response_dict)}")

        return response_dict


# ==register in API
api.add_resource(Filter, "/api/v1/filters")


# / enriched model
# ====/api/v1/digital_twin/enrichment===================================================================================
# ==derive responses
class EnrichedDigitalTwinModel(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /digital_twin/enrichment with dict arguments")
        # validate inputs validate_request
        try:
            validated_args = validate_request(API_GET_ENRICHED_TWIN_MODEL_SCHEME, json_data)
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}

        progress_tracker = ProgressTrackerDataTransformation()

        # input was valid - build response
        build_digital_twin_enrichment_response(get_digital_twin_updated_func=get_digital_twin_updated_func,
                                               args=validated_args, multi_scenarios_handler=multi_scenarios_handler,
                                               digital_twin_file_path=DIGITAL_TWIN_FILE_PATH,
                                               project_path=PROJECT_PATH, progress_tracker=progress_tracker,
                                               data_source_name=DATA_SOURCE_MODEL_NAME)

        colored_print(f"[API]\tEnrichment response: '""' \n")
        return "OK"


# ==register in API
api.add_resource(EnrichedDigitalTwinModel, "/api/v1/digital_twin/enrichment")


def get_unpacked_args_kpi(args):
    start_time = get_datetime(args["dateStart"])
    end_time = get_datetime(args["dateEnd"])
    order_ids_list = args["orders"]
    product_ids_list = args["products"]
    process_ids_list = args["processes"]
    resource_ids_list = args["resources"]
    scenario = args["scenario"]
    if "units" in args:  # not a must argument
        units: dict = args["units"]
    else:
        units = {}
    return start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, units, scenario


# ====/api/v1/orders====================================================================================================
# ==derive responses
class Orders(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /orders with dict arguments: {json_data}")
        # validate inputs validate_request
        try:
            validated_args = validate_request(API_GET_ORDERS_SCHEME, json_data)
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}
        # input was valid - build response
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, units, scenario = (
            get_unpacked_args_kpi(validated_args))
        scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario)

        colored_print(f"[API] Build orders response with args: {validated_args}")
        coroutine = build_orders_response(
            start_time=start_time, end_time=end_time, order_ids_list=order_ids_list, product_ids_list=product_ids_list,
            process_ids_list=process_ids_list, resource_ids_list=resource_ids_list, units=units,
            scenario_handler=scenario_handler)
        response_object = loop.run_until_complete(coroutine)
        response = response_object.get_response_dict()

        colored_print(f"[API]\tOrders response: {response} \n")
        return response


# ==register in API
api.add_resource(Orders, "/api/v1/orders")


# ====/api/v1/orders/summary============================================================================================
# ==derive responses
class OrdersSummary(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /orders/summary with dict arguments: {json_data}")
        # validate inputs validate_request
        try:
            validated_args = validate_request(API_GET_ORDERS_SCHEME, json_data)
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}
        # input was valid - build response
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, units, scenario = (
            get_unpacked_args_kpi(validated_args))
        scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario)

        colored_print(f"[API] Build orders summary response with args: {validated_args}")
        coroutine = build_orders_summary_response(
            start_time=start_time, end_time=end_time, order_ids_list=order_ids_list, product_ids_list=product_ids_list,
            process_ids_list=process_ids_list, resource_ids_list=resource_ids_list, units=units,
            scenario_handler=scenario_handler)
        response_object = loop.run_until_complete(coroutine)
        response = response_object.get_response_dict()

        colored_print(f"[API]\tOrders summary response: {response} \n")
        return response


# ==register in API
api.add_resource(OrdersSummary, "/api/v1/orders/summary")


# ====/api/v1/products==================================================================================================
# ==derive responses
class Products(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /products with dict arguments: {json_data}")
        # validate inputs validate_request
        try:
            validated_args = validate_request(API_GET_PRODUCTS_SCHEME, json_data)
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}
        # input was valid - build response
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, units, scenario = (
            get_unpacked_args_kpi(validated_args))
        scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario)

        colored_print(f"[API] Build products response with args: {validated_args}")
        coroutine = build_products_response(
            start_time=start_time, end_time=end_time, order_ids_list=order_ids_list, product_ids_list=product_ids_list,
            process_ids_list=process_ids_list, resource_ids_list=resource_ids_list, units=units,
            scenario_handler=scenario_handler)
        response_object = loop.run_until_complete(coroutine)
        response = response_object.get_response_dict()

        colored_print(f"[API]\tProducts response: {response} \n")
        return response


# ==register in API
api.add_resource(Products, "/api/v1/products")


# ====/api/v1/products/summary==========================================================================================
# ==derive responses
class ProductsSummary(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /products/summary with dict arguments: {json_data}")
        # validate inputs validate_request
        try:
            validated_args = validate_request(API_GET_PRODUCTS_SCHEME, json_data)
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}
        # input was valid - build response
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, units, scenario = (
            get_unpacked_args_kpi(validated_args))
        scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario)

        colored_print(f"[API] Build products summary response with args: {validated_args}")
        coroutine = build_products_summary_response(
            start_time=start_time, end_time=end_time, order_ids_list=order_ids_list, product_ids_list=product_ids_list,
            process_ids_list=process_ids_list, resource_ids_list=resource_ids_list, units=units,
            scenario_handler=scenario_handler)
        response_object = loop.run_until_complete(coroutine)
        response = response_object.get_response_dict()

        colored_print(f"[API]\tProducts summary response: {response} \n")
        return response


# ==register in API
api.add_resource(ProductsSummary, "/api/v1/products/summary")


# ====/api/v1/processes=================================================================================================
# ==derive responses
class Processes(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /processes with dict arguments: {json_data}")
        # validate inputs validate_request
        try:
            validated_args = validate_request(API_GET_PROCESSES_SCHEME, json_data)
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}
        # input was valid - build response
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, units, scenario = (
            get_unpacked_args_kpi(validated_args))
        scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario)

        colored_print(f"[API] Build processes response with args: {validated_args}")
        coroutine = build_processes_response(
            start_time=start_time, end_time=end_time, order_ids_list=order_ids_list, product_ids_list=product_ids_list,
            process_ids_list=process_ids_list, resource_ids_list=resource_ids_list, units=units,
            scenario_handler=scenario_handler)
        response_object = loop.run_until_complete(coroutine)
        response = response_object.get_response_dict()

        colored_print(f"[API]\tProcesses response: {response} \n")
        return response


# ==register in API
api.add_resource(Processes, "/api/v1/processes")


# ====/api/v1/processes/summary=========================================================================================
# ==derive responses
class ProcessesSummary(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /processes summary with dict arguments: {json_data}")
        # validate inputs validate_request
        try:
            validated_args = validate_request(API_GET_PROCESSES_SCHEME, json_data)
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}
        # input was valid - build response
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, units, scenario = (
            get_unpacked_args_kpi(validated_args))
        scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario)

        colored_print(f"[API] Build processes summary response with args: {validated_args}")
        coroutine = build_processes_summary_response(
            start_time=start_time, end_time=end_time, order_ids_list=order_ids_list, product_ids_list=product_ids_list,
            process_ids_list=process_ids_list, resource_ids_list=resource_ids_list, units=units,
            scenario_handler=scenario_handler)
        response_object = loop.run_until_complete(coroutine)
        response = response_object.get_response_dict()

        colored_print(f"[API]\tProcesses summary response: {response} \n")
        return response


# ==register in API
api.add_resource(ProcessesSummary, "/api/v1/processes/summary")


# ====/api/v1/resources=================================================================================================
# ==derive responses
class Resources(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /resources with dict arguments: {json_data}")
        # validate inputs validate_request
        try:
            validated_args = validate_request(API_GET_RESOURCES_SCHEME, json_data)
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}
        # input was valid - build response
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, units, scenario = (
            get_unpacked_args_kpi(validated_args))
        scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario)

        colored_print(f"[API] Build resources response with args: {validated_args}")
        coroutine = build_resources_response(
            start_time=start_time, end_time=end_time, order_ids_list=order_ids_list, product_ids_list=product_ids_list,
            process_ids_list=process_ids_list, resource_ids_list=resource_ids_list, units=units,
            scenario_handler=scenario_handler)
        response_object = loop.run_until_complete(coroutine)
        response = response_object.get_response_dict()

        colored_print(f"[API]\tResources response: {response} \n")
        return response


# ==register in API
api.add_resource(Resources, "/api/v1/resources")


# ====/api/v1/resources/summary=========================================================================================
# ==derive responses
class ResourcesSummary(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /resources summary with dict arguments: {json_data}")
        # validate inputs validate_request
        try:
            validated_args = validate_request(API_GET_RESOURCES_SCHEME, json_data)
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}
        # input was valid - build response
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, units, scenario = (
            get_unpacked_args_kpi(validated_args))
        scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario)

        colored_print(f"[API] Build resources summary response with args: {validated_args}")
        coroutine = build_resources_summary_response(
            start_time=start_time, end_time=end_time, order_ids_list=order_ids_list, product_ids_list=product_ids_list,
            process_ids_list=process_ids_list, resource_ids_list=resource_ids_list, units=units,
            scenario_handler=scenario_handler)
        response_object = loop.run_until_complete(coroutine)
        response = response_object.get_response_dict()

        colored_print(f"[API]\tResources summary response: {response} \n")
        return response


# ==register in API
api.add_resource(ResourcesSummary, "/api/v1/resources/summary")


# CHARTS/ VISUALIZATION

def get_unpacked_args_charts(args):
    start_time = get_datetime(args["dateStart"])
    end_time = get_datetime(args["dateEnd"])
    order_ids_list = args["orders"]
    product_ids_list = args["products"]
    process_ids_list = args["processes"]
    resource_ids_list = args["resources"]
    bin_size = args["bin"]
    scenario = args["scenario"]
    return start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, bin_size, \
        scenario

# ====/api/v1/resources=================================================================================================
# ==derive responses
class UtilizationChartResources(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /resources/chart/utilization with dict arguments: {json_data}")
        # validate inputs validate_request
        try:
            validated_args = validate_request(API_GET_UTILIZATION_CHART_RESOURCES_SCHEME, json_data)
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}
        # input was valid - build response
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        start_time, end_time, order_ids_list, product_ids_list, process_ids_list, resource_ids_list, bin_size, scenario = \
            get_unpacked_args_charts(validated_args)  # resource_type="ALL"
        scenario_handler = multi_scenarios_handler.get_scenario_handler_by_name(scenario)

        colored_print(f"[API] Build resources utilization chart response with args: {validated_args}")

        coroutine = build_resources_utilization_chart_response(
            start_time=start_time, end_time=end_time, order_ids_list=order_ids_list, product_ids_list=product_ids_list,
            process_ids_list=process_ids_list, resource_ids_list=resource_ids_list, bin_size=bin_size,
            scenario_handler=scenario_handler)
        response_object = loop.run_until_complete(coroutine)
        response = response_object.get_response_dict()

        colored_print(f"[API]\tResources chart utilization response: to long \n")
        return response


# ==register in API
api.add_resource(UtilizationChartResources, "/api/v1/resources/chart/utilization")


# ====/api/v1/resources=================================================================================================
def get_lead_time_chart_response(json_data, view):
    raise NotImplementedError

    response_object = build_lead_time_chart_response(validated_args, multi_scenarios_handler, view)
    response = response_object.get_response_dict()

    return response


# ==derive responses
class LeadTimeChartOrders(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /orders/chart/lead_time with dict arguments: {json_data}")
        # validate inputs validate_request

        response = get_lead_time_chart_response(json_data, view="ORDER")
        colored_print(f"[API]\tOrders chart lead_time response: {response} \n")
        return response


# ==register in API
api.add_resource(LeadTimeChartOrders, "/api/v1/orders/chart/lead_time")


class LeadTimeChartProducts(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /products/chart/lead_time with dict arguments: {json_data}")
        # validate inputs validate_request

        response = get_lead_time_chart_response(json_data, view="PRODUCT")
        colored_print(f"[API]\tProducts chart lead_time response: {response} \n")
        return response


# ==register in API
api.add_resource(LeadTimeChartProducts, "/api/v1/products/chart/lead_time")


class LeadTimeChartProcesses(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /processes/chart/lead_time with dict arguments: {json_data}")
        # validate inputs validate_request

        response = get_lead_time_chart_response(json_data, view="PROCESS")
        colored_print(f"[API]\tProcesses chart lead_time response: {response} \n")
        return response


# ==register in API
api.add_resource(LeadTimeChartProcesses, "/api/v1/processes/chart/lead_time")


class LeadTimeChartResources(Resource):

    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /resources/chart/lead_time with dict arguments: {json_data}")
        # validate inputs validate_request

        response = get_lead_time_chart_response(json_data, view="RESOURCE")
        colored_print(f"[API]\tProcesses chart lead_time response: {response} \n")
        return response


# ==register in API
api.add_resource(LeadTimeChartResources, "/api/v1/resources/chart/lead_time")


# Data Transformation

# current_state_available
# ====/api/v1/digital_twin/current_state_available======================================================================
# ==derive responses
class CurrentStateAvailableModel(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /digital_twin/current_state_available with dict arguments")
        # validate inputs validate_request
        try:
            validate_request(API_GET_CURRENT_STATE_AVAILABLE, json_data)
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}

        # input was valid - build response
        current_state_available = (
            build_digital_twin_current_state_available(current_state_available=API_SET_CURRENT_STATE_AVAILABLE))

        colored_print(f"[API]\tCurrent state available response: '{current_state_available}' \n")
        return current_state_available


# ==register in API
api.add_resource(CurrentStateAvailableModel, "/api/v1/digital_twin/current_state_available")


# SIMULATION

# determines the files in the folder
# ====/api/v1/simulation/model_paths====================================================================================
# ==derive responses
class SimulationParameters(Resource):
    def post(self):
        # get payload of request
        colored_print(f"[API] Got POST-Request on /simulation model paths with dict arguments")
        # validate inputs validate_request

        # input was valid - build response
        response_object = (
            build_simulation_model_paths_response(folder_path=PROJECT_PATH,
                                                  simulation_parameters=API_SET_SIMULATION_PARAMETER_SCENARIO_SCHEME))
        response = response_object.get_response_dict()

        colored_print(f"[API]\tSimulation model paths response: {response} \n")
        return response


# ==register in API
api.add_resource(SimulationParameters, "/api/v1/simulation/model_paths")


class StartSimulation(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()["params"]
        colored_print(f"[API] Got POST-Request on /start simulation with dict arguments: {json_data}")
        # validate inputs validate_request
        try:
            validated_args = validate_request(API_GET_SIMULATION_SCHEME, json_data)  # API_GET_RESOURCES_SCHEME
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}
        # input was valid - build response

        progress_tracker = ProgressTrackerSimulation()
        scenario = (
            build_simulation_response(simulation_func=simulation_func, args=validated_args,
                                      multi_scenarios_handler=multi_scenarios_handler,
                                      scenario_model_paths=scenario_model_paths,
                                      progress_tracker=progress_tracker, project_path=PROJECT_PATH))

        colored_print(f"[API]\tStart simulation response: '{scenario}' \n")
        return scenario


# ==register in API
api.add_resource(StartSimulation, "/api/v1/simulation/start")


# ====/api/v1/scenario/overview=========================================================================================
# ==derive responses
class ScenarioOverview(Resource):
    def post(self):
        # get payload of request
        colored_print(f"[API] Got POST-Request on /scenario overview")
        # validate inputs validate_request

        # input was valid - build response
        response_object = (
            build_scenario_overview_response(multi_scenarios_handler=multi_scenarios_handler,
                                             export_path=API_SCENARIO_EXPORT_PATH_SCHEME))
        response = response_object.get_response_dict()

        colored_print(f"[API]\tScenario overview paths response: {response} \n")
        return response


# ==register in API
api.add_resource(ScenarioOverview, "/api/v1/scenario/overview")


# ====/api/v1/scenario/export=========================================================================================
# ==derive responses
class ScenarioExport(Resource):
    def post(self):
        # get payload of request
        json_data = request.get_json()
        colored_print(f"[API] Got POST-Request on /scenario export with dict arguments: {json_data}")
        # validate inputs validate_request
        try:
            validated_args = validate_request(API_POST_SCENARIO_EXPORT_SCHEME, json_data)  # API_GET_RESOURCES_SCHEME
        except TypeError as e:
            # input document was invalid - return error (including details) as response
            return {"error": e}
        # input was valid - build response
        zip_file_path = build_scenario_export_response(validated_args,
                                                       multi_scenarios_handler=multi_scenarios_handler)

        colored_print(f"[API]\tScenario export zip file path response: {zip_file_path} \n")
        return send_file(zip_file_path, as_attachment=True)


# ==register in API
api.add_resource(ScenarioExport, "/api/v1/scenario/export")


class ProgressTrackerDataTransformation:

    def announce(self, msg):
        sendMessage(msg)


class ProgressTrackerSimulation:

    def announce(self, msg):
        sendMessage(msg)


def event_stream(client_queue):
    # send initial data
    yield 'data: Connected at: %s\n\n' % time.ctime()

    while True:
        message = client_queue.get()
        yield 'data: %s\n\n' % message


def sendMessage(message):
    try:
        for client_queue in listeners["SIMULATION"]:
            client_queue.put(message)
    except:
        pass


@app.route('/api/v1/simulation/progress')
def stream():
    client_queue = queue.Queue()
    listeners.setdefault("SIMULATION", []).append(client_queue)
    return Response(event_stream(client_queue), mimetype="text/event-stream")


# @app.route('/api/v1/data_transformation/progress')
# def stream():
#     client_queue = queue.Queue()
#     listeners.setdefault("DATA_TRANSFORMATION", []).append(client_queue)
#     return Response(event_stream(client_queue), mimetype="text/event-stream")


listeners = {}

# ====start server======================================================================================================
if __name__ == "__main__":
    pass
