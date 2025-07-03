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

Module to generate random customers and orders.

@contact persons: Adrian Freiter
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Union, Type, Optional

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.repository_services.deserialization.order_types import OrderType, ProductClassSelection
from ofact.twin.state_model.sales import Order

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.state_model.sales import Customer, FeatureCluster, Feature
    from ofact.twin.state_model.processes import ValueAddedProcess, WorkOrder
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
    type_hints: dict = {"OrderPoolGenerator": {"Number of Orders": int,
                                               "Order Type": None,#list[*list(OrderType)],
                                               "Product Class Selection":  None,#list[*list(ProductClassSelection)],
                                               "Earliest Order Date": datetime,
                                               "Latest Order Date": datetime,
                                               "Earliest Delivery Date Planned": datetime,
                                               "Latest Delivery Date Planned": datetime}}
    default_values: dict = {"OrderPoolGenerator": {"Number of Orders": 1,
                                                   "Order Type": list(OrderType)[0],
                                                   "Product Class Selection": list(ProductClassSelection)[0],
                                                   "Earliest Order Date": datetime(1970, 1, 1),
                                                   "Latest Order Date": datetime(1970, 1, 1),
                                                   "Earliest Delivery Date Planned": datetime(1970, 1, 1),
                                                   "Latest Delivery Date Planned": datetime(1970, 1, 1)}}

    @classmethod
    def get_init_parameter_type_hints(cls, digital_twin_class: Union[str, Type]) -> Optional[dict[str, object]]:
        return cls.type_hints[digital_twin_class]

    @classmethod
    def get_init_parameter_default_values(cls, digital_twin_class: Union[str, Type]) -> Optional[dict[str, object]]:
        return cls.default_values[digital_twin_class]

    def __init__(self, customers: list[Customer], feature_clusters: list[FeatureCluster],
                 features: list[Feature], value_added_processes: list[ValueAddedProcess],):
        """
        The OrderGenerator can generate sales_orders.

        Parameters
        ----------
        customers: all customers that can be chosen for the orders
        feature_clusters: all feature clusters (from each one feature have to be chosen)
        """

        self.customers = customers
        self.feature_clusters = feature_clusters
        self.feature_clusters_features = {}
        for feature in features:
            self.feature_clusters_features.setdefault(feature.feature_cluster,
                                                      []).append(feature)

        feature_process_mapping = defaultdict(list)
        for process in value_added_processes:
            feature_process_mapping[process.feature].append(process)  # Testing required
        self.processes_by_feature = feature_process_mapping

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
                  start_datetime: datetime = datetime.now(),
                  order_type=OrderType.PRODUCT_CONFIGURATOR,
                  order_lead_time: timedelta = None,
                  product_class_mix: dict[EntityType, float] = {},
                  external_identifications={},
                  domain_specific_attributes={}):
        """
        Generate a new order and return them.

        Parameters
        ----------

        order_type: type of generator (for details see the description on the OrderType class
        (above))
        - PRODUCT_CONFIGURATOR: select one feature of a feature_cluster
        - SHOPPING_BASKET: a feature can be selected zero, one, or more times for one order

        start_datetime: used for the order_date and the delivery_date_planned
        order_lead_time: lead time of the order used for the delivery_date_planned determination
        product_class_mix: specify the probability a product class is chosen
        external_identifications: usable for specification of the order
        domain_specific_attributes: for specification of the order

        Returns
        -------
        new_order: a digital_twin order object
        """
        product_classes = self.get_product_classes(product_class_selection_probabilities=product_class_mix)
        features_requested = self.get_random_product_by_product_classes(product_classes=product_classes,
                                                                      generation_type=order_type)

        product_classes = self._adapt_product_class(product_classes=product_classes,
                                                    features_requested=features_requested)

        price = self.get_price(features_requested)
        customer = self.get_random_customer()
        order_date = self.get_order_date(start_datetime, features_requested)
        delivery_date_requested = self.get_delivery_date_requested(order_date, features_requested)
        delivery_date_planned = self.get_delivery_date_planned(order_date, features_requested, order_lead_time)
        urgent = self.get_urgent()
        new_order = Order(identification=None,
                          product_classes=product_classes,
                          features_requested=features_requested,
                          features_completed=[],
                          customer=customer,
                          products=None,
                          price=price,
                          order_date=order_date,
                          delivery_date_requested=delivery_date_requested,
                          delivery_date_planned=delivery_date_planned,
                          delivery_date_actual=None,
                          urgent=urgent,
                          external_identifications=external_identifications,
                          domain_specific_attributes=domain_specific_attributes)

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

    def get_delivery_date_planned(self, order_date, features_requested, order_lead_time: timedelta = None):
        if order_lead_time is None:
            order_lead_time = timedelta(seconds=len(features_requested) * 250)
        delivery_date_planned = order_date + order_lead_time
        return delivery_date_planned

    def get_urgent(self):
        urgent = 0
        return urgent

    def get_product_classes(self, product_class_selection_probabilities: dict[EntityType, float] = None):
        if product_class_selection_probabilities is None:
            product_class = random.choice(list(self.product_class_feature_cluster_feature_mapping.keys()))
        else:
            product_class = random.choice(list(product_class_selection_probabilities.keys()))
        product_classes = [product_class]  # ToDo: When to choose more than one product class
        return product_classes

    def get_random_product_by_product_classes(self, product_classes: list[Union[PartType, EntityType]], generation_type):
        """
        Generation of a randomized bike on the basis of the feature clusters and features
        - FeatureCluster:Components of a bike: frame, Gabel, etc.
        - Feature: concrete design of a component: Trofeo 5, Carbon Öldruckstoßdämpfer, etc.
        and the determination of the order price

        Parameters
        ----------
        product_classes: list of product classes
        generation_type: type of generator (for details see the description on the OrderType class (above))
        - PRODUCT_CONFIGURATOR: select one feature of a feature_cluster
        - SHOPPING_BASKET: a feature can be selected zero, one, or more times for one order

        Returns
        -------
        features_requested
        """
        features_requested = []
        for product_class in product_classes:
            feature_cluster_feature_mapping = self.product_class_feature_cluster_feature_mapping[product_class]

            # weighted feature selection based on the chosen/ set generation_behaviour
            features_requested_product_class = self.feature_generator[generation_type](feature_cluster_feature_mapping)
            features_requested.extend(features_requested_product_class)

        return features_requested

    def _adapt_product_class(self, product_classes: list[EntityType | PartType], features_requested: list[Feature]):
        """
        The product classes are adapted to the features requested
        - if the product class does not match the features requested,
        the product class is adapted to the features requested
        """
        adapted_product_classes = []
        for product_class in product_classes:
            adapted_product_class = None
            for feature in features_requested:
                processes: list[ValueAddedProcess] = self.processes_by_feature[feature]

                for process in processes:
                    if process.successors:
                        continue

                    possible_output_entity_types: list[tuple[EntityType, int]] = (
                        process.get_possible_output_entity_types())

                    for possible_output_entity_type in possible_output_entity_types:
                        if (product_class.check_entity_type_match_lower(possible_output_entity_type[0]) and
                                possible_output_entity_type[0] != product_class):
                            adapted_product_class = possible_output_entity_type[0]
                            break

                    if adapted_product_class is not None:
                        break
                if adapted_product_class is not None:
                    break
                else:
                    adapted_product_class = product_class

            adapted_product_classes.append(adapted_product_class)

        return adapted_product_classes

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

    def get_orders(self, amount: int, order_type: OrderType, start_datetime: datetime = datetime.now(),
                   order_lead_time: datetime = None, product_class_mix: dict[EntityType, float] = None) -> list[Order]:
        """
        Generation of a randomized order "amount" times.

        Parameters
        ----------
        amount: the amount of orders that should be generated
        start_datetime: the order_date attribute of the first order.
        order_lead_time: used for the calculation of the delivery_date_planned of the order
        product_class_mix: map the probability to choose a product class (value) to the product class (key) itself
        """

        start_datetime_l = [start_datetime]
        for i in range(amount):
            # adjust start_date
            start_datetime += timedelta(seconds=random.randint(500, 700))
            start_datetime_l.append(start_datetime)  # ToDo: more standardized way?

        new_orders = [self.get_order(start_datetime_l[i], order_type=order_type, order_lead_time=order_lead_time,
                                     product_class_mix=product_class_mix)
                      for i in range(amount)]
        return new_orders

    def get_orders_pending(self, amount: int, order_type: OrderType, start_datetime=datetime.now()) -> list[Order]:
        """
        Get generated orders
        Returns
        -------
        orders_pending: a list of pending_orders
        """
        new_orders = self.get_orders(amount, order_type, start_datetime)
        return new_orders


class OrderPoolCharacteristicsDeriver:

    def __init__(self, orders: list[Order], feature_clusters: list[FeatureCluster]):
        """
        The class is used to derive characteristics/parameters from the order pool,
        that can be used to generate new orders.

        Parameters
        ----------
        orders: a list of orders that form the order pool (from this pool, the parameters are derived)
        feature_clusters: a list of feature_clusters (used for parameter derivation - e.g., order_type)
        """

        self.orders: list[Order] = orders
        self.feature_clusters: list[FeatureCluster] = feature_clusters

    def _determine_order_type(self):
        """
        Can an order pool be mixed with different order types?
        This method assumes an order pool where each order has the same order type.
        """
        samples_to_test_max = 2

        order_type = OrderType.SHOPPING_BASKET
        for order in self.orders:
            order_feature_cluster = [feature.feature_cluster
                                     for feature in order.features_requested]
            if len(order_feature_cluster) > len(set(order_feature_cluster)):
                order_type = OrderType.PRODUCT_CONFIGURATOR
            else:
                feature_clusters_product_class = [feature_cluster
                                                  for feature_cluster in self.feature_clusters
                                                  for order_product_class in order.product_classes
                                                  if feature_cluster.product_class == order_product_class]
                if len(feature_clusters_product_class) > len(set(order_feature_cluster)):
                    order_type = OrderType.PRODUCT_CONFIGURATOR

            if samples_to_test_max == 0:
                break
            else:
                samples_to_test_max -= 1

        return order_type

    def _get_order_product_class_mix(self):
        product_classes_count = {}
        for order in self.orders:
            for order_product_class in order.product_classes:
                if order_product_class in product_classes_count:
                    product_classes_count[order_product_class] += 1
                else:
                    product_classes_count[order_product_class] = 1

        product_class_amount = sum(list(product_classes_count.values()))
        product_class_share_mix = {product_class: product_class_count / product_class_amount
                                   for product_class, product_class_count in product_classes_count.items()}

        return product_class_share_mix

    def get_earliest_order_date(self):
        order_dates_min = min([order.order_date
                               for order in self.orders
                               if order.order_date])
        return order_dates_min

    def get_latest_order_date(self):
        order_dates_max = max([order.order_date
                               for order in self.orders
                               if order.order_date])
        return order_dates_max

    def _determine_orders_per_hour(self):
        earliest_order_date = self.get_earliest_order_date()
        latest_order_date = self.get_latest_order_date()

        time_span = latest_order_date - earliest_order_date
        orders_per_hour = len(self.orders) / time_span
        return orders_per_hour

    def _determine_planned_order_lead_times(self):
        """Based on the planned delivery date and the order date."""
        order_lead_times = [order.delivery_date_planned - order.order_date
                            for order in self.orders
                            if order.delivery_date_planned and order.order_date]
        order_lead_time_mean = sum(order_lead_times) / len(order_lead_times)

        return order_lead_time_mean


# ToDo: Provide a method set that transit the characteristics to the current or other circumstances
#  (e.g., the earliest_order_date)


class DigitalTwinOrderGenerator:

    def __init__(self, digital_twin_model: StateModel, work_in_progress, factor=1):
        """
        The digital twin order generator adds orders to the state model of the digital twin.
        It additionally enables the order pool creation based on the current factory capacities ().
        """

        self.digital_twin_model = digital_twin_model
        customer = self.digital_twin_model.get_customers()
        feature_clusters = self.digital_twin_model.get_feature_clusters()
        features = self.digital_twin_model.get_features()

        resource_time_share_required = self.determine_factory_capacity_required()

        value_added_processes = self.digital_twin_model.get_value_added_processes()
        self.order_generator = OrderGenerator(customers=customer, feature_clusters=feature_clusters,
                                              features=features, value_added_processes=value_added_processes)

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
        random_orders = [self.order_generator.get_order()
                         for i in range(5)]

        order_value_added_process_time_mean = self.digital_twin_model.get_estimated_order_lead_time_mean(random_orders)

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

    def add_orders(self, amount, start_datetime=datetime.now(), order_type=OrderType.PRODUCT_CONFIGURATOR):
        orders = self.order_generator.get_orders_pending(amount=amount, start_datetime=start_datetime,
                                                         order_type=order_type)

        self.digital_twin_model.add_orders(orders)


if __name__ == '__main__':
    from ofact.planning_services.model_generation.persistence import get_state_model_file_path, deserialize_state_model
    from projects.bicycle_world.settings import PROJECT_PATH

    stat_model_file_name = "base_wo_material_supply.pkl"
    stat_model_file_path =  get_state_model_file_path(project_path=PROJECT_PATH,
                                                      state_model_file_name=stat_model_file_name,
                                                     path_to_model="scenarios/current/models/twin/")
    state_model = deserialize_state_model(source_file_path=stat_model_file_path, persistence_format="pkl")

    dt_order_generator = DigitalTwinOrderGenerator(digital_twin_model=state_model,
                                                   work_in_progress=19)
    dt_order_generator.add_orders(amount=5, order_type=OrderType.PRODUCT_CONFIGURATOR)
