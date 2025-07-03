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

Used to import the state model of the twin with generating elements.
"""

# Imports Part 1: Standard Imports
from pathlib import Path

# Imports Part 2: PIP Imports
import pandas as pd

# Imports Part 3: Project Imports
from ofact.planning_services.model_generation.sales_generator import CustomerGenerator, OrderGenerator
from ofact.twin.repository_services.deserialization.order_types import OrderType
from ofact.twin.repository_services.deserialization.static_state_model import (
    StaticStateModelDeserialization, get_object_dicts, update_feature_weights, get_order_generation_from_excel_df)
from ofact.twin.state_model.sales import Customer


class StaticStateModelGenerator(StaticStateModelDeserialization):

    def __init__(self, path, CUSTOMER_GENERATION_FROM_EXCEL=False, ORDER_GENERATION_FROM_EXCEL=False,
                 ORDER_TYPE=OrderType.PRODUCT_CONFIGURATOR, CUSTOMER_AMOUNT=5, ORDER_AMOUNT=5):
        """
        Used to load/ create the digital twin from the Excel file.

        Parameters
        ----------
        path: a reference path to the Excel file
        CUSTOMER_GENERATION_FROM_EXCEL: customer can be generated from the customer generator or from the Excel file
        ORDER_GENERATION_FROM_EXCEL: order can be generated from the order generator or from the Excel file
        CUSTOMER_AMOUNT: used if customers are generated to determine the amount of customers generated
        ORDER_AMOUNT: used if orders are generated to determine the amount of orders generated
        """

        self.CUSTOMER_GENERATION_FROM_EXCEL = CUSTOMER_GENERATION_FROM_EXCEL
        self.ORDER_GENERATION_FROM_EXCEL = ORDER_GENERATION_FROM_EXCEL
        self.customer_amount = CUSTOMER_AMOUNT
        self.order_amount = ORDER_AMOUNT

        super().__init__(path=path, ORDER_TYPE=ORDER_TYPE)

    def create_customer_from_excel(self):

        if self.CUSTOMER_GENERATION_FROM_EXCEL:
            self.customer_df = self._get_df(sheet_name="Customer", skiprows=None)
            self.customer_df.dropna(how='all', inplace=True)

            self.customer_df["name"] = self.customer_df["pre_name"] + " " + self.customer_df["last_name"]
            del self.customer_df["pre_name"]
            del self.customer_df["last_name"]
            mapping_class = self.mappings["MappingCustomer"]
            self.customer_objects, _ = self.object_instantiation.load_dict(object_df=self.customer_df,
                                                                           mapping_class=mapping_class)
        else:
            customers_df = self._get_df(sheet_name='CustomerGeneration', index_col=[], skiprows=None)
            customers_df.dropna(how='all', inplace=True)
            pre_names = customers_df["pre_names"].to_list()
            last_names = customers_df["last_names"].to_list()
            location = [location
                        for location in customers_df["locations"].to_list()
                        if type(location) == str]
            providers = [provider
                         for provider in customers_df["providers"].to_list()
                         if type(provider) == str]

            customers = CustomerGenerator(pre_names=pre_names, last_names=last_names, location=location,
                                          providers=providers)
            customer_dict = customers.get_customer(self.customer_amount)

            # instantiate the customers
            customers = [Customer(**customer_attributes)
                         for customer_attributes in customer_dict]

            self.customer_objects = {("Customer", str(customer.identification)): customer
                                     for customer in customers}

    def create_orders_from_excel(self):
        feature_objects = get_object_dicts(self.sales_objects, "Feature")
        features = list(feature_objects.values())
        update_feature_weights(features, generation_type=self.ORDER_TYPE)

        if self.ORDER_GENERATION_FROM_EXCEL:

            self.order_df = get_order_generation_from_excel_df(path=self.xlsx_content)

            mapping_class = self.mappings["MappingOrders"]
            self.order_objects, _ = self.object_instantiation.load_dict(object_df=self.order_df,
                                                                        mapping_class=mapping_class,
                                                                        input_objects=[self.customer_objects,
                                                                                       self.sales_objects])
        else:
            # orders
            feature_clusters_objects = get_object_dicts(self.sales_objects, "FeatureCluster")
            feature_clusters = list(feature_clusters_objects.values())
            customers = list(get_object_dicts(self.customer_objects, "Customer").values())
            value_added_processes = list(get_object_dicts(self.process_objects, "ValueAddedProcess").values())
            # order_pool
            order_generator = OrderGenerator(customers=customers, feature_clusters=feature_clusters, features=features,
                                             value_added_processes=value_added_processes)
            order_pool = order_generator.get_orders_pending(amount=self.order_amount, order_type=self.ORDER_TYPE)

            self.order_objects = {("Order", str(order.identification)): order
                                  for order in order_pool}
