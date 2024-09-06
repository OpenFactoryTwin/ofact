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

@contact persons: Adrian Freiter
"""

# Imports Part 1: Standard Imports
from pathlib import Path

# Imports Part 2: PIP Imports
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.state_model.model import StateModel
from ofact.twin.state_model.sales import Customer
from ofact.twin.repository_services.interface.importer.order_types import OrderType
from ofact.twin.repository_services.interface.importer.twin_importer import StaticModelImporter, get_object_dicts, \
    update_feature_weights, get_order_generation_from_excel_df
from ofact.planning_services.model_generation.sales_generator import (CustomerGenerator, OrderGenerator)


class StaticModelGenerator(StaticModelImporter):

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
            self.customer_df = pd.read_excel(self.xlsx_content, index_col=[0, 1], sheet_name="Customer")
            self.customer_df.dropna(how='all', inplace=True)

            self.customer_df["name"] = self.customer_df["pre_name"] + " " + self.customer_df["last_name"]
            del self.customer_df["pre_name"]
            del self.customer_df["last_name"]
            mapping_class = type(self).mappings["MappingCustomer"]
            self.customer_objects, _ = self.object_instantiation.load_dict(object_df=self.customer_df,
                                                                           mapping_class=mapping_class)
        else:
            customers_df = pd.read_excel(self.xlsx_content, sheet_name='CustomerGeneration')
            customers_df.dropna(how='all', inplace=True)
            pre_names = customers_df["pre_names"].to_list()
            last_names = customers_df["last_names"].to_list()
            location = [location for location in customers_df["locations"].to_list() if type(location) == str]
            providers = [provider for provider in customers_df["providers"].to_list() if type(provider) == str]

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

            mapping_class = type(self).mappings["MappingOrder"]
            self.order_objects, _ = self.object_instantiation.load_dict(object_df=self.order_df,
                                                                        mapping_class=mapping_class,
                                                                        input_objects=[self.customer_objects,
                                                                                       self.sales_objects])
        else:
            # orders
            feature_clusters_objects = get_object_dicts(self.sales_objects, "FeatureCluster")
            feature_clusters = list(feature_clusters_objects.values())
            customers = list(get_object_dicts(self.customer_objects, "Customer").values())

            # order_pool
            order_generator = OrderGenerator(customers=customers, feature_clusters=feature_clusters, features=features,
                                             order_type=self.ORDER_TYPE)
            order_pool = order_generator.get_orders_pending(amount=self.order_amount)

            self.order_objects = {("Order", str(order.identification)): order
                                  for order in order_pool}


def get_digital_twin(project_path, digital_twin_file_name=None, path_to_model="models/twin/",
                     digital_twin_file_path=None, digital_twin_objects_class=None, pickle_=True,
                     customer_generation_from_excel=False, order_generation_from_excel=False,
                     customer_amount=0, order_amount=0,
                     order_type=OrderType.SHOPPING_BASKET):
    """Return the digital_twin object from the pickle or xlsx file"""

    file_path, pickle_ = _get_digital_twin_file_path(digital_twin_file_path, digital_twin_file_name,
                                                     project_path, path_to_model, pickle_)

    if not pickle_:
        digital_twin_model = (
            get_digital_twin_model_from_excel(digital_twin_objects_class, file_path, customer_generation_from_excel,
                                              order_generation_from_excel, customer_amount, order_amount,
                                              order_type))

    else:
        digital_twin_model = StateModel.from_pickle(digital_twin_file_path=file_path)

    return digital_twin_model


def get_digital_twin_model_from_excel(static_model_importer_class, file_path, customer_generation_from_excel,
                                      order_generation_from_excel, customer_amount, order_amount,
                                      order_type):
    if static_model_importer_class is None:
        static_model_importer_class = StaticModelGenerator

    static_model_importer = \
        static_model_importer_class(file_path,
                                   CUSTOMER_GENERATION_FROM_EXCEL=customer_generation_from_excel,
                                   ORDER_GENERATION_FROM_EXCEL=order_generation_from_excel,
                                   CUSTOMER_AMOUNT=customer_amount, ORDER_AMOUNT=order_amount,
                                   ORDER_TYPE=order_type)

    digital_twin_model = static_model_importer.get_digital_twin()
    if isinstance(file_path, str):
        file_path = file_path.split(".")[0] + ".pkl"
    else:
        file_path = str(file_path).split(".")[0] + ".pkl"
    digital_twin_model.to_pickle(digital_twin_file_path=file_path)

    return digital_twin_model


def _get_digital_twin_file_path(digital_twin_file_path, digital_twin_file_name, project_path, path_to_model, pickle_):
    """Return the digital twin file path"""

    if digital_twin_file_path is None:
        if digital_twin_file_name is None:
            raise Exception("Please provide either the digital_twin_file_path or the digital_twin_file_name")

        path = str(digital_twin_file_name).split('.')[0]
        if pickle_:
            digital_twin_file_name = f"{path}.pkl"
        else:
            digital_twin_file_name = f"{path}.xlsx"

        # digital_twin initialisation
        file_path = Path(str(project_path), path_to_model + digital_twin_file_name)

    else:
        pickle_ = True  # should be a pickle ...
        file_path = digital_twin_file_path

    return file_path, pickle_
