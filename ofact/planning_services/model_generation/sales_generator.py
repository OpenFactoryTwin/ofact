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

Module to generate randomly customers and orders.

@contact persons: Adrian Freiter
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Union

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.state_model.sales import Order
from ofact.twin.repository_services.interface.importer.order_types import OrderType

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.state_model.sales import Customer, FeatureCluster, Feature
    from ofact.twin.state_model.entities import PartType, EntityType


class CustomerGenerator:

    def __init__(self, pre_names: list[str], last_names: list[str], location: list[str], providers: list[str]):
        """
        The Customer generator can instantiate sales_customers.

        Parameters
        ----------
        pre_names: Pre-names (used as customer pre_name)
        last_names: last_names (used as customer last_name)
        location: (home) location (used as customer location)
        providers: e_mail_address provider (used for the customer e_mail_address)
        """
        self.pre_names = pre_names
        self.last_names = last_names
        self.location = location
        self.providers = providers

    def get_customer(self, amount) -> list[dict]:
        """
        Create the given amount of customers parameters (dict)

        Parameters
        ----------
        amount: determines the amount of customer parameter created

        Returns
        -------
        a list of customer parameter dicts
        """
        iterations = np.linspace(1, amount, amount)
        customer_parameters = []
        if amount == 0:
            return customer_parameters

        if not (self.pre_names and self.last_names):
            raise Exception("No name choices available")
        if not (self.location):
            raise Exception("No location choices available")
        if not (self.providers):
            raise Exception("No provider choices available")

        for it in iterations:
            customer_dict = {}
            customer_dict["identification"] = None
            customer_dict["name"] = random.choice(self.pre_names) + " " + random.choice(self.last_names)
            customer_dict["location"] = random.choice(self.location)
            customer_dict["e_mail_address"] = customer_dict['name'].split(' ')[0] + "." + \
                                              customer_dict['name'].split(' ')[1] + '@' + \
                                              random.choice(self.providers)
            customer_parameters.append(customer_dict)

        return customer_parameters


class OrderGenerator:

    def __init__(self, customers: list[Customer], feature_clusters: list[FeatureCluster],
                 features: list[Feature], order_type=OrderType.PRODUCT_CONFIGURATOR):
        """
        The OrderGenerator can generate sales_orders.

        Parameters
        ----------
        customers: all customers that can be chosen for the orders
        feature_clusters: all feature cluster (from each one feature have to be chosen)
        order_type: type of generator (for details see the description on the OrderType class
        (above))
        - PRODUCT_CONFIGURATOR: select one feature of a feature_cluster
        - SHOPPING_BASKET: a feature can be selected zero, one, or more times for one order
        """

        self.customers = customers
        self.feature_clusters = feature_clusters
        self.feature_clusters_features = {}
        for feature in features:
            self.feature_clusters_features.setdefault(feature.feature_cluster,
                                                      []).append(feature)
        self.order_type = order_type
        self.feature_generator = \
            {OrderType.PRODUCT_CONFIGURATOR: self._get_random_features_product_configurator,
             OrderType.SHOPPING_BASKET: self._get_random_features_shopping_basket}
        self.product_class_feature_cluster_feature_mapping = self.get_initial_mapping()
        self.orders_pending = []

        self.realising_cycle = None

    def get_initial_mapping(self) -> dict[EntityType, dict[FeatureCluster, np.array([[Feature, float]])]]:
        """
        Map features with probabilities to feature_cluster and the feature clusters to product_classes

        Returns
        -------
        features with probabilities to feature_cluster and the feature clusters to product_classes
        """
        # map all features to their respective feature_clusters
        product_class_feature_cluster_feature_mapping = {}
        # e.g. {product_class: {feature_cluster: [feature1, feature2, ...]}}
        product_classes = {feature_cluster.product_class
                           for feature_cluster in self.feature_clusters}
        for product_class in product_classes:
            product_class_feature_cluster_feature_mapping[product_class] = {}
            for feature_cluster in self.feature_clusters:
                if feature_cluster.product_class == product_class:
                    product_class_feature_cluster_feature_mapping[product_class][feature_cluster] = (
                        self.feature_clusters_features[feature_cluster])

        return product_class_feature_cluster_feature_mapping

    def set_release_cycles(self, estimated_releasing_cycle):
        self.realising_cycle = estimated_releasing_cycle

    def get_order(self,
                  start_datetime=datetime.now(),
                  external_identifications={},
                  domain_specific_attributes={}):
        """
        Generate a new order and return them.

        Parameters
        ----------
        start_datetime: used for the order_date and the delivery_date_planned
        external_identifications: for specification of the order usable
        domain_specific_attributes: for specification of the order usable

        Returns
        -------
        new_order: a digital_twin order object
        """
        product_class = self.get_random_product_class()
        features_requested = self.get_random_product_by_product_class(product_class=product_class,
                                                                      generation_type=self.order_type)

        price = self.get_price(features_requested)
        customer = self.get_random_customer()
        order_date = self.get_order_date(start_datetime, features_requested)
        delivery_date_requested = self.get_delivery_date_requested(order_date, features_requested)
        delivery_date_planned = self.get_delivery_date_planned(order_date, features_requested)
        urgent = self.get_urgent()
        new_order = Order(identification=None,
                          product_class=product_class,
                          features_requested=features_requested,
                          features_completed=[],
                          customer=customer,
                          product=None,
                          price=price,
                          order_date=order_date,
                          delivery_date_requested=delivery_date_requested,
                          delivery_date_planned=delivery_date_planned,
                          delivery_date_actual=None,
                          urgent=urgent,
                          external_identifications=external_identifications,
                          domain_specific_attributes=domain_specific_attributes
                          )

        return new_order

    def get_random_customer(self):
        if self.customers:
            customer = random.choice(self.customers)
        else:
            customer = None
        return customer

    def get_order_date(self, start_datetime, features_requested):
        order_date = self.random_date(start=start_datetime,
                                      end=start_datetime + timedelta(seconds=len(features_requested)),
                                      prop=random.random())
        return order_date

    def get_delivery_date_requested(self, order_date, features_requested):
        return None

    def get_delivery_date_planned(self, order_date, features_requested):
        delivery_date_planned = order_date + timedelta(seconds=len(features_requested) * 250)
        return delivery_date_planned

    def get_urgent(self):
        urgent = 0
        return urgent

    def get_random_product_class(self):
        product_class = random.choice(list(self.product_class_feature_cluster_feature_mapping.keys()))
        return product_class

    def get_random_product_by_product_class(self, product_class: Union[PartType, EntityType], generation_type):
        """
        Generation of a randomized bike on the basis of the feature clusters and features
        - featureCluster (Components of a bike: frame, Gabel, etc.)
        - feature        (concrete design of a component: Trofeo 5, Carbon Öldruckstoßdämpfer, etc.)
        and the determination of the order price
        """

        feature_cluster_feature_mapping = self.product_class_feature_cluster_feature_mapping[product_class]

        # weighted feature selection based on the chosen/ set generation_behaviour
        features_requested = self.feature_generator[generation_type](feature_cluster_feature_mapping)

        return features_requested

    def _get_random_features_product_configurator(self, feature_cluster_feature_mapping):
        features_requested = \
            [random.choices(population=features,
                            weights=[feature.get_selection_probability() for feature in features])[0]
             for feature_cluster, features in feature_cluster_feature_mapping.items()]

        return features_requested

    def _get_random_features_shopping_basket(self, feature_cluster_feature_mapping):
        # assumption: probability is given by distribution (for example normal distribution)
        features_with_amounts = \
            [(feature, round(feature.get_selection_probability()))
             for feature_cluster, features in feature_cluster_feature_mapping.items()
             for feature in features]
        features_requested = [feature
                              for feature, amount in features_with_amounts
                              if amount > 0
                              for i in range(amount)]

        return features_requested

    def random_date(self, start, end, prop=random.random()):
        """
        Get a time at a proportion of a range of two formatted times.
        Parameters
        ----------
        start: start and end should be strings specifying times formatted in the given format (strftime-style),
        giving an interval [start, end].
        end: end times stamp
        prop: specifies how a proportion of the interval to be taken after start.

        Returns
        -------
        time will be in the specified format.
        """

        ptime = start + timedelta(seconds=(prop * (end - start)).seconds)

        return ptime

    def get_price(self, features_requested):
        price = sum([feature.get_price()
                     for feature in features_requested])
        return price

    def get_new_orders(self, amount, start_datetime: datetime = datetime.now()) -> list[Order]:
        """
        Generation of a randomized order "amount" times.
        """

        start_datetime_l = [start_datetime]
        for i in range(amount):
            # adjust start_date
            start_datetime += timedelta(seconds=random.randint(500, 700))
            start_datetime_l.append(start_datetime)

        new_orders = [self.get_order(start_datetime_l[i])
                      for i in range(amount)]
        return new_orders

    def get_orders_pending(self, amount: int, start_datetime=datetime.now()) -> list[Order]:
        """
        Get generated orders
        Returns
        -------
        orders_pending: a list of pending_orders
        """
        new_orders = self.get_new_orders(amount, start_datetime)
        return new_orders


class DigitalTwinOrderGenerator:

    def __init__(self, digital_twin_model: StateModel, order_type: OrderType, work_in_progress,
                 factor=1):
        self.digital_twin_model = digital_twin_model
        customer = self.digital_twin_model.get_customers()
        feature_clusters = self.digital_twin_model.get_feature_clusters()
        features = self.digital_twin_model.get_features()

        resource_time_share_required = self.determine_factory_capacity_required()

        self.order_generator = OrderGenerator(customers=customer, feature_clusters=feature_clusters,
                                              features=features, order_type=order_type)

        order_value_added_process_time_mean = self.get_order_value_added_process_time_mean()

        estimated_releasing_cycle = self.get_estimated_releasing_cycle(resource_time_share_required,
                                                                       order_value_added_process_time_mean,
                                                                       work_in_progress=work_in_progress,
                                                                       factor=factor)

        self.order_generator.set_release_cycles(estimated_releasing_cycle)

    def determine_factory_capacity_required(self):
        value_added_processes = self.digital_twin_model.get_value_added_processes()

        selection_probabilities_features = self.determine_feature_selection_probabilities()

        resource_time_share_needed = {}
        for value_added_process in value_added_processes:
            feature = value_added_process.feature
            selection_probability = selection_probabilities_features[feature]

            lead_time_required = value_added_process.get_estimated_process_lead_time()
            resource_groups = value_added_process.get_resource_groups()

            resource_group_alternatives = len(resource_groups)
            for resource_group in resource_groups:
                for resource_et in resource_group.resources:
                    all_resources = self.digital_twin_model.get_all_resources_by_entity_types([resource_et])
                    resources = [elem for lst in list(all_resources.values()) for elem in lst]

                    share = len(resources) * resource_group_alternatives
                    for resource in resources:
                        resource_time_share_needed.setdefault(resource, 0)

                        # more efficient resources are assumed to have more capacity
                        performance = resource.get_expected_performance()
                        resource_lead_time_required = ((lead_time_required * performance * selection_probability) /
                                                       share)

                        resource_time_share_needed[resource] += resource_lead_time_required

        return resource_time_share_needed

    def get_order_value_added_process_time_mean(self):

        feature_value_added_processes = self.digital_twin_model.get_feature_process_mapper()

        random_orders = [self.order_generator.get_order()
                         for i in range(5)]

        # feature value_added_process required mapping
        random_orders_value_added_processes_required = (
            {order: [value_added_processes
                     for feature in order.features_requested
                     for value_added_processes in feature_value_added_processes[feature]]
             for order in random_orders})

        # value added process order lead_time mapping
        random_orders_lead_time_required = (
            {order: sum([value_added_process.get_estimated_process_lead_time()
                         for value_added_process in value_added_processes])
             for order, value_added_processes in random_orders_value_added_processes_required.items()})

        order_lead_times = list(random_orders_lead_time_required.values())
        order_value_added_process_time_mean: float = sum(order_lead_times) / len(order_lead_times)

        return order_value_added_process_time_mean

    def get_estimated_releasing_cycle(self, resource_time_share_required, order_value_added_process_time_mean,
                                      work_in_progress, factor):
        resources_time_required_lst = list(resource_time_share_required.values())
        resources_time_required_in_system = sum(resources_time_required_lst)
        orders_lead_time_in_system = work_in_progress * order_value_added_process_time_mean
        releasing_cycle_rough = (order_value_added_process_time_mean *
                                 (orders_lead_time_in_system / resources_time_required_in_system))
        estimated_releasing_cycle: float = releasing_cycle_rough * factor

        return estimated_releasing_cycle

    def determine_feature_selection_probabilities(self):
        selection_probabilities_features = {feature: feature.get_expected_selection_probability()
                                            for feature in self.digital_twin_model.get_features()}

        return selection_probabilities_features

    def add_orders(self, amount, start_datetime=datetime.now()):
        orders = self.order_generator.get_orders_pending(amount=amount, start_datetime=start_datetime)

        self.digital_twin_model.add_orders(orders)


if __name__ == '__main__':
    from ofact.planning_services.model_generation.twin_generator import get_digital_twin
    from projects.bicycle_world.settings import PROJECT_PATH

    digital_twin_file_name = "base_wo_material_supply.pkl"
    digital_twin_model = get_digital_twin(PROJECT_PATH,
                                          digital_twin_file_name,
                                          path_to_model="scenarios/current/models/twin/",
                                          pickle_=True)

    dt_order_generator = DigitalTwinOrderGenerator(digital_twin_model=digital_twin_model,
                                                   order_type=OrderType.PRODUCT_CONFIGURATOR,
                                                   work_in_progress=19)
    dt_order_generator.add_orders(amount=5)
