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

Encapsulates all classes concerning the sales view of the digital twin model

Classes:
    Customer: Customer that buys products
    FeatureCluster: A product is described by Features organized in feature clusters
    Feature: Features of a product from a customer perspective (e.g., sport drive)
    Order: the concrete customer order that specifies the product class the features and the delivery date

@contact persons: Christian Schwede & Adrian Freiter
@last update: 14.05.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from copy import copy
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Union

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.state_model.basic_elements import DigitalTwinObject, DynamicDigitalTwinObject
from ofact.twin.state_model.serialization import Serializable
from ofact.twin.state_model.entities import EntityType, Part, PartType
from ofact.twin.state_model.probabilities import SingleValueDistribution
from ofact.twin.state_model.helpers.helpers import convert_to_datetime

if TYPE_CHECKING:
    from ofact.twin.state_model.probabilities import ProbabilityDistribution
    from ofact.twin.state_model.processes import ProcessExecution


class Customer(DigitalTwinObject, Serializable):

    def __init__(self,
                 name: str,
                 location: Optional[str] = None,
                 e_mail_address: Optional[str] = None,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Customers buy products.

        Parameters
        ----------
        name: Name of the customer
        location: location of the customer, may be used to consolidate products with the same location
        e_mail_address: e-mail address from the customer
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.name: str = name
        self.location: Optional[str] = location
        self.e_mail_address: Optional[str] = e_mail_address

    def copy(self):
        customer_copy = super(Customer, self).copy()

        return customer_copy

    def completely_filled(self):
        not_completely_filled_attributes = []
        if self.identification is None:
            not_completely_filled_attributes.append("identification")
        if not isinstance(self.name, str):
            not_completely_filled_attributes.append("name")
        if not isinstance(self.location, str):
            not_completely_filled_attributes.append("location")
        if not isinstance(self.e_mail_address, str):
            not_completely_filled_attributes.append("e_mail_address")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes

    def __str__(self):
        return (f"Customer with ID '{self.identification}' and name '{self.name}'; "
                f"'{self.location}', '{self.e_mail_address}'")


class FeatureCluster(DigitalTwinObject, Serializable):
    drop_before_serialization = []
    further_serializable = ['product_class']

    def __init__(self,
                 name: str,
                 product_class: Union[PartType, EntityType],
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Feature cluster belongs to a product class and consists of mutual exclusive features.
        For each cluster of a product class, one feature has to be selected to specify the product.

        Parameters
        ----------
        name: Name of the feature cluster
        product_class: entity type that specifies the product
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.name: str = name
        self.product_class: Union[PartType, EntityType] = product_class

    def copy(self):
        feature_cluster_copy = super(FeatureCluster, self).copy()

        return feature_cluster_copy

    def dict_serialize(self, serialize_private: bool = True,
                       deactivate_id_filter: bool = False,
                       use_label: bool = False,
                       use_label_for_situated_in: bool = True) -> Union[dict | str]:
        """
        Creates a dict representation of the object.
        Also serializes objects in attributes by calling the function on them.

        Parameters
        ----------
        serialize_private: Whether to serialize the private attributes.
            Defaults to True.
        deactivate_id_filter: Whether to check if an obj has already been serialized.
            Then the static model id is used. Defaults to False.
        use_label_for_situated_in: No functionality.
        use_label: Whether to represent this object by its label.

        Returns
        -------
        object_dict: The dict representation of the entity type or the static model id.
        """
        object_dict = super().dict_serialize(serialize_private=serialize_private,
                                             deactivate_id_filter=deactivate_id_filter,
                                             use_label=use_label,
                                             use_label_for_situated_in=use_label_for_situated_in)
        if isinstance(object_dict, str):
            return object_dict
        for key, value in object_dict.items():
            if key in self.further_serializable and value is not None:
                object_dict[key] = value.dict_serialize()

        # Check if attributes are missing
        Serializable.warn_if_attributes_are_missing(list(self.__dict__.keys()),
                                                    ignore=self.drop_before_serialization,
                                                    dictionary=object_dict,
                                                    use_ignore=True)

        return object_dict

    def __str__(self):
        return (f"Customer with ID '{self.identification}' and name '{self.name}'; "
                f"'{self.product_class}'")


class Feature(DigitalTwinObject, Serializable):
    drop_before_serialization = []
    further_serializable = ['feature_cluster', 'selection_probability_distribution']

    def __init__(self,
                 name: str,
                 feature_cluster: FeatureCluster,
                 price: Optional[float],
                 selection_probability_distribution: ProbabilityDistribution = SingleValueDistribution(1),
                 is_not_chosen_option: bool = False,
                 identification: Optional[int] = None,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        Feature is a selectable feature for a product class in an order

        Parameters
        ----------
        name: Name of the feature cluster
        is_not_chosen_option: True if the feature is not chosen
        feature_cluster: the corresponding feature cluster
        price: the market price of the feature, sum of all feature prices is the product price
        selection_probability_distribution: the probability that the feature is chosen
        - for product configurator the probability should be sum up to one over all possible features of
        a feature cluster
        - for shopping basket the probability is represented by probability_distribution, because 0 to n elements
        of a feature could be chosen, represented by the selection_probability_distribution
        """
        super().__init__(identification=identification, external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.name: str = name
        self.is_not_chosen_option: bool = is_not_chosen_option
        self.feature_cluster: FeatureCluster = feature_cluster
        self.price: Optional[float] = price
        self.selection_probability_distribution: ProbabilityDistribution = selection_probability_distribution

    def __str__(self):
        return (f"Feature with ID '{self.identification}' and name '{self.name}'; "
                f"'{self.is_not_chosen_option}', '{self.feature_cluster}', '{self.price}', "
                f"'{self.selection_probability_distribution}'")

    def copy(self):
        feature_copy = super(Feature, self).copy()

        return feature_copy

    def dict_serialize(self, serialize_private: bool = True,
                       deactivate_id_filter: bool = False,
                       use_label: bool = False,
                       use_label_for_situated_in: bool = True) -> Union[dict | str]:
        """
        Creates a dict representation of the object.
        Also serialize an object in attributes by calling the function on them.

        Parameters
        ----------
        serialize_private: Whether to serialize the private attributes.
        Defaults to True.
        use_label: If set to true, only the static_model_id is returned for this object.
        Defaults to False.
        deactivate_id_filter: Whether to check if an obj has already been serialized.
        Then the static model id is used. Defaults to False.
        use_label_for_situated_in: Whether to use only the label for the situated_in attribute.

        Returns
        -------
        object_dict: The dict representation of the entity type or the static model id.
        """
        if use_label:
            return self.get_static_model_id()
        object_dict = super().dict_serialize(serialize_private=serialize_private,
                                             deactivate_id_filter=deactivate_id_filter,
                                             use_label=use_label,
                                             use_label_for_situated_in=use_label_for_situated_in)
        if isinstance(object_dict, str):
            return object_dict

        for key, value in object_dict.items():
            if key in self.further_serializable and value is not None:
                object_dict[key] = value.dict_serialize()

        # Check if attributes are missing
        Serializable.warn_if_attributes_are_missing(list(self.__dict__.keys()),
                                                    ignore=self.drop_before_serialization,
                                                    dictionary=object_dict,
                                                    use_ignore=True)
        return object_dict

    def get_expected_selection_probability(self):
        """
        Returns
        -------
        the expected selection probability from the selection_probability distribution,
        which is equal or higher as zero
        """
        expected_selection_probability = self.selection_probability_distribution.get_expected_value()
        return expected_selection_probability

    def get_selection_probability(self):
        """
        Returns
        -------
        a selection probability sampled from the selection_probability distribution,
        which is equal or higher as zero
        """
        selection_probability = self.selection_probability_distribution.get_random_number()
        selection_probability_non_negative = max(0, selection_probability)
        return selection_probability_non_negative

    def get_price(self) -> float:
        """Return a price value as float (None is converted to 0)"""
        if self.price is not None:
            price = self.price
        else:
            price = 0
        return price


class Order(DynamicDigitalTwinObject, Serializable):
    drop_before_serialization = ['dynamic_attributes']

    def __init__(self,
                 identifier: Optional[str] = None,
                 product_class: Optional[Union[EntityType, PartType]] = None,
                 features_requested: Optional[list[Feature]] = None,
                 customer: Optional[Customer] = None,
                 order_date: Optional[Union[int, datetime]] = None,
                 release_date: Optional[Union[int, datetime]] = None,
                 delivery_date_requested: Optional[Union[int, datetime]] = None,
                 delivery_date_planned: Optional[Union[int, datetime]] = None,
                 delivery_date_actual: Optional[Union[int, datetime]] = None,
                 urgent: Optional[int] = None,
                 features_completed: Optional[list[Feature]] = None,
                 product: Optional[Part] = None,
                 price: Optional[float] = 0,
                 process_executions: Optional[list[ProcessExecution]] = None,
                 identification: Optional[int] = None,
                 process_execution: Optional[ProcessExecution] = None,
                 current_time: datetime = datetime.min,
                 external_identifications: Optional[dict[object, list[object]]] = None,
                 domain_specific_attributes: Optional[dict[str, Optional[object]]] = None):
        """
        (Sales-)Order is an order from a customer.
        The order object is associated with a specific product (part),
        specified by the product_class (entity_type/ part_type).
        In addition, some datetime values are provided, that describe the time chronological progress of the order.

        Parameters
        ----------
        identifier: next to the identification that is digital twin model intern,
        the identifier is used as external identifier for the order (in general, the order ID should be more
        human readable, since the identifier is displayed on the analytics dashboard)
        product_class: part or entity type that specifies the product processed by the order
        features_requested: features requested by the customer (one per each feature cluster)
        features_completed: features of teh requested feature that have already been completed in production
        customer: Customer who bought the product
        product: physical product as a concrete part
        order_date: Date in seconds the product was ordered
        release_date: Date the order was released to process
        delivery_date_requested: requested delivery date from the customer
        (should be the same as the planned one, if possible)
        delivery_date_planned: planned delivery date
        delivery_date_actual: actual delivery date
        urgent: normally a bool value (0: not urgent; 1: urgent),
        with values higher 1 can be used for more urgent orders
        price: the market price of the feature, sum of all feature prices is the product price

        attribute feature_process_execution_match: match the process_executions with the features of the order.
        process_executions used for the order but cannot be mapped directly to a feature are mapped
        to the feature "None".
        this allows determining the progress of the order in a more detailed way.
        """
        if features_completed is None:
            features_completed = []
        self.features_completed: list[Feature] = features_completed

        super().__init__(identification=identification, process_execution=process_execution, current_time=current_time,
                         external_identifications=external_identifications,
                         domain_specific_attributes=domain_specific_attributes)
        self.identifier = identifier

        self.product_class: Optional[EntityType, PartType] = product_class

        if features_requested is None:
            features_requested = []
        self.features_requested: list[Feature] = features_requested

        self.price: Optional[float] = price

        if not isinstance(customer, Customer):
            customer = None
        self.customer: Optional[Customer] = customer
        self.product: Optional[Part] = product

        if isinstance(order_date, str):
            order_date = convert_to_datetime(order_date)
        self.order_date: Optional[int, datetime] = order_date

        if isinstance(release_date, str):
            release_date = convert_to_datetime(order_date)
        self.release_date: Optional[int, datetime] = release_date

        if isinstance(delivery_date_requested, str):
            delivery_date_requested = convert_to_datetime(delivery_date_requested)
        self.delivery_date_requested: Optional[int, datetime] = delivery_date_requested

        if isinstance(delivery_date_planned, str):
            delivery_date_planned = convert_to_datetime(delivery_date_planned)
        self.delivery_date_planned: Optional[int, datetime] = delivery_date_planned

        if isinstance(delivery_date_actual, str):
            delivery_date_actual = convert_to_datetime(delivery_date_actual)
        self.delivery_date_actual: Optional[int, datetime] = delivery_date_actual

        self.urgent: Optional[int] = urgent

        self.feature_process_execution_match: dict[Feature: list[ProcessExecution]] = {}

    def __str__(self):
        return (f"Order with ID '{self.identification}' from customer '{self.get_customer_name()}'; "
                f"'{self.price}', '{self.get_product_name()}', '{self.get_product_class_name()}', "
                f"'{self.get_feature_requested_names()}', '{self.get_feature_completed_names()}', "
                f"'{self.order_date}', '{self.release_date}', '{self.delivery_date_requested}', "
                f"'{self.delivery_date_planned}', '{self.delivery_date_actual}', '{self.urgent}', "
                f"'{self.feature_process_execution_match}'")

    def copy(self):
        order_copy: Order = super(Order, self).copy()
        order_copy.features_completed = order_copy.features_completed.copy()
        order_copy.features_requested = order_copy.features_requested.copy()
        order_copy.feature_process_execution_match = order_copy.feature_process_execution_match.copy()

        return order_copy

    def get_identifier(self):
        return self.identifier

    def get_progress_status(self):
        """
        Determine the status information according to the progress of the order
        Based on the available order datetime (release date / delivery date) and the features
        (requested and completed)
        """
        number_features_requested = len(self.features_completed)
        number_features_completed = len(self.features_completed)

        if self.delivery_date_actual:
            progress_status = 100

        elif not number_features_completed:
            if self.release_date:
                # case: order started, but no features finished - providing small progress of max 1 %
                progress_of_one_feature = int(100 / (number_features_requested * 2))
                progress_status = min(1, progress_of_one_feature)

            else:
                progress_status = 0

        else:
            progress_status = ((number_features_completed + 1e-9) /
                               (number_features_requested + number_features_completed + 1e-9))

        return progress_status

    def get_feature_requested_names(self):
        return [feature.name
                for feature in self.features_requested]

    def get_feature_completed_names(self):
        feature_completed_names = [feature.name
                                   for feature in self.features_completed]
        return feature_completed_names

    def get_product_name(self):

        if self.product is not None:
            product_name = self.product.name
        else:
            product_name = None
        return product_name

    def get_customer_name(self):
        if self.customer is not None:
            customer_name = self.customer.name
        else:
            customer_name = None
        return customer_name

    def get_product_class_name(self):

        if self.product_class is not None:
            product_class_name = self.product_class.name
        else:
            product_class_name = None
        return product_class_name

    def get_value_added_process_executions(self):
        """Returns the value_added_process_executions associated with the order"""

        process_executions_dict = self.feature_process_execution_match.copy()
        if None in process_executions_dict:
            process_executions_dict.pop(None)
        value_added_process_executions_nested = list(process_executions_dict.values())
        value_added_process_executions_completed = [process_execution
                                                    for process_execution_lst in value_added_process_executions_nested
                                                    for process_execution in process_execution_lst]

        return value_added_process_executions_completed

    def get_process_executions(self):
        """Returns the process_executions associated with the order"""
        process_executions_nested = list(self.feature_process_execution_match.values())
        process_executions_completed = [process_execution
                                                    for process_execution_lst in process_executions_nested
                                                    for process_execution in process_execution_lst]

        return process_executions_completed

    def get_sorted_process_executions(self):
        """Sort the process executions according their time stamps (executed start time stamp)"""
        process_executions = self.get_process_executions()

        process_executions_sorted = sorted(process_executions,
                                           key=lambda process_execution: (process_execution.executed_start_time,
                                                                          process_execution.get_process_lead_time()))

        return process_executions_sorted

    def get_features_with_value_added_processes(self):
        """
        Get the features of the order that need to be completed through value added processes.
        The other features are marked as done.
        """

        features_with_value_added_processes = [feature
                                               for feature in self.features_requested
                                               if not feature.is_not_chosen_option]
        features_without_value_added_processes = [feature
                                                  for feature in self.features_requested
                                                  if feature.is_not_chosen_option]

        self.mark_features_without_value_added_processes_as_done(features_without_value_added_processes)

        return features_with_value_added_processes

    def release(self, release_date: Optional[datetime] = None):
        """
        Release the order.
        Tasks: Set the release date of the order

        Parameters
        ----------
        release_date: The release date of the order
        """
        if release_date is None:
            release_date = datetime.now()

        self.release_date = release_date

    def complete(self, delivery_date_actual: Optional[datetime] = None):
        """
        Complete the order.
        Tasks: Set the delivery date actual of the order

        Parameters
        ----------
        delivery_date_actual: The delivery date actual of the order
        """
        if self.features_requested:
            print(f"Warning: The order {self.identifier} is completed, but not all features have been completed")

        if delivery_date_actual is None:
            delivery_date_actual = datetime.now()

        self.delivery_date_actual = delivery_date_actual

    def add_process_execution(self, process_execution: ProcessExecution):
        """Add a process_execution that is associated with the order"""
        self.feature_process_execution_match.setdefault(None,
                                                        []).append(process_execution)

    def mark_features_without_value_added_processes_as_done(self, features_without_value_added_processes):
        """
        Used to mark features in the current_order as done which do not need any value_added_process.
        For example, the features no_ring describes only that no ring is chosen.

        Parameters
        ----------
        features_without_value_added_processes: a list of features that did not have any value_added_processes
        """
        self.features_requested = \
            [feature_requested for feature_requested in self.features_requested
             if feature_requested not in features_without_value_added_processes]
        for feature_without_value_added_process in features_without_value_added_processes:
            self.complete_feature(feature_without_value_added_process)

    def complete_feature(self, feature_completed: Feature,
                         process_executions: Optional[list[ProcessExecution] | ProcessExecution] = None,
                         sequence_already_ensured: bool = False):
        """
        Complete the feature based on all process_executions that are responsible for the feature completion

        Parameters
        ----------
        feature_completed: feature that should be marked as completed
        process_executions: process_executions that is responsible for the change
        sequence_already_ensured: used in the dynamic attributes for higher performance by assuming
        that the execution of the process is performed sequentially in a time chronological order
        """
        if process_executions is None:
            process_executions = []
        elif not isinstance(process_executions, list):
            process_executions = [process_executions]
        process_executions: list[ProcessExecution]

        feature_founded = [feature_requested for feature_requested in self.features_requested
                           if feature_requested.identification == feature_completed.identification]

        # exceptions
        if len(feature_founded) == 0:
            feature_already_completed = [feature for feature in self.features_completed
                                         if feature.identification == feature_completed.identification]
            if not len(feature_already_completed) > 0:
                raise ValueError(f"[{self.__class__.__name__}] complete_feature from the order '{self.identification}' "
                                 f"/ '{self.external_identifications}' cannot be conducted "
                                 f"because the feature '{feature_completed.name}' is not requested")

        # complete feature
        feature = feature_founded[0]
        self.features_requested.remove(feature)
        # .append leads to a not understandable behavior
        # because copy is related to the list - the feature remains the same object
        self.features_completed = copy(self.features_completed) + [feature]
        self.match_feature_process_executions(feature, process_executions)

        # update the dynamic attribute

        if process_executions:
            # find the last process_execution
            last_process_execution_idx = np.argmax([process_execution.executed_end_time
                                                    for process_execution in process_executions])
            current_time = process_executions[last_process_execution_idx].get_latest_available_time_stamp()
            self.update_attributes(process_execution=process_executions[last_process_execution_idx],
                                   current_time=current_time,
                                   features_completed=feature, sequence_already_ensured=sequence_already_ensured,
                                   change_type="ADD")

    def update_feature(self, feature: Feature, process_executions: Optional[list[ProcessExecution]] = None):
        """
        Complete the feature based on all process_executions that are responsible for the feature completion

        Parameters
        ----------
        feature: to the feature the process_executions are mapped
        process_executions: process_executions that are responsible for the change
        """
        if feature in self.features_requested or feature in self.features_completed:
            self.match_feature_process_executions(feature, process_executions)

    def get_release_date_from_process_executions(self):
        """
        Determine the release date based on the process executions (chronological first process execution)
        Note: this could be a incorrect if a process is executed before the order is released
        but give an insight if the "real" release date is not set.
        """
        order_process_executions = self.get_sorted_process_executions()
        if order_process_executions:
            # assuming that the first element in the process_executions has the earliest executed start time
            first_process_execution: ProcessExecution = order_process_executions[0]
            if first_process_execution.executed_start_time is not None:
                return first_process_execution.executed_start_time

        return None

    def get_delivery_date_actual_from_process_executions(self):
        """Determine the delivery_date_actual based on the process executions if no feature is requested anymore"""
        order_process_executions = self.get_sorted_process_executions()
        if order_process_executions and not self.features_requested:
            # assuming that the first element in the process_executions has the earliest executed start time
            first_process_execution: ProcessExecution = order_process_executions[0]
            if first_process_execution.executed_start_time is not None:
                return first_process_execution.executed_start_time

        return None

    def add_product(self, part: Part):
        """Add a part as product"""
        if part.entity_type.check_entity_type_match(self.product_class):
            self.product = part
        else:
            raise ValueError(f"[{self.__class__.__name__}] add_product cannot be conducted "
                             f"because the part has a wrong entity_type")

    def add_delivery_date_actual(self, delivery_date):
        """Add the delivery_date_actual"""
        # check if the order is already completed
        if not self.features_requested:
            self.delivery_date_actual = delivery_date
        else:
            raise ValueError(f"[{self.__class__.__name__}] add_delivery_date_actual cannot be conducted "
                             f"because not all features_requested are executed")

    def match_feature_process_executions(self, feature: Feature, process_executions: list[ProcessExecution]):
        """Map a new process_execution to the feature (needed for the kpi calculation)"""
        # ToDo: the order request more than one equal feature - the assignment of the process_executions
        #  to the right feature is currently not implemented, which means: all process_executions
        #  of different features (which are equal) are mapped to the same feature

        self.feature_process_execution_match.setdefault(feature,
                                                        []).extend(process_executions)

    def completely_filled(self) -> (bool, list):
        """
        Check if the Order object can be filled.
        It does not mean that the order object is filled, it can only review the minimum requirements
        Like orders should have at least one feature that should be completed and so on.
        It also doesn't determine if the order is completed/ end.
        """

        not_completely_filled_attributes = []
        if self.identification is None:
            not_completely_filled_attributes.append("identification")
        if not isinstance(self.customer, Customer):
            not_completely_filled_attributes.append("customer")
        if not isinstance(self.product, Part):
            not_completely_filled_attributes.append("product")
        if not isinstance(self.product_class, EntityType):
            not_completely_filled_attributes.append("product_class")
        if not isinstance(self.price, float):
            not_completely_filled_attributes.append("price")
        if not isinstance(self.features_requested, list):
            not_completely_filled_attributes.append("features_requested")
        elif not self.features_requested:
            not_completely_filled_attributes.append("features_requested")
        if not isinstance(self.features_completed, list):
            not_completely_filled_attributes.append("features_completed")
        if not (isinstance(self.order_date, datetime) or isinstance(self.order_date, int)):
            not_completely_filled_attributes.append("order_date")
        if not (isinstance(self.delivery_date_requested, datetime) or isinstance(self.delivery_date_requested, int)):
            not_completely_filled_attributes.append("delivery_date_requested")
        if not (isinstance(self.delivery_date_planned, datetime) or isinstance(self.delivery_date_planned, int)):
            not_completely_filled_attributes.append("delivery_date_planned")
        if not (isinstance(self.delivery_date_actual, datetime) or isinstance(self.delivery_date_actual, int)):
            not_completely_filled_attributes.append("delivery_date_actual")

        if not_completely_filled_attributes:
            completely_filled = False
        else:
            completely_filled = True
        return completely_filled, not_completely_filled_attributes

    def dict_serialize(self, serialize_private: bool = True,
                       deactivate_id_filter: bool = False,
                       use_label: bool = False,
                       use_label_for_situated_in: bool = True) -> Union[dict | str]:
        """
        Converts the object to a json serializable dict.

        Parameters
        ----------
        serialize_private: Whether to serialize the private attributes.
            Defaults to True.
        deactivate_id_filter: Whether to check if an obj has already been serialized.
            Then the static model id is used. Defaults to False.
        use_label: Whether to use the static model id as representation. Defaults to False.
        use_label_for_situated_in: Whether to use the static model id as representation for the situated_in
            attribute. Defaults to False.

        Returns
        -------
        object_dict: The dict representation of the entity type or the static model id.
        """

        object_dict = super().dict_serialize(serialize_private=serialize_private,
                                             deactivate_id_filter=deactivate_id_filter,
                                             use_label=use_label,
                                             use_label_for_situated_in=use_label_for_situated_in)
        further_serializable = ['feature', 'customer', 'product']
        further_serializable_list = ['features_completed',
                                     'features_requested',
                                     '_process_executions',
                                     'feature_process_execution_match']
        datetime_serializable = ['order_date',
                                 'release_date',
                                 'delivery_date_requested',
                                 'delivery_date_planned',
                                 'delivery_date_actual']

        for key, value in object_dict.items():
            if key in further_serializable and hasattr(value, 'dict_serialize'):
                object_dict[key] = value.dict_serialize()
            elif key in further_serializable_list and value is not None:
                if key == 'feature_process_execution_match':
                    object_dict[key] = Serializable.serialize_dict(value)
                else:
                    object_dict[key] = Serializable.serialize_list(value)
            elif key in datetime_serializable and isinstance(value, datetime):
                object_dict[key] = value.timestamp()

        # Check if attributes are missing
        Serializable.warn_if_attributes_are_missing(list(self.__dict__.keys()),
                                                    ignore=self.drop_before_serialization,
                                                    dictionary=object_dict,
                                                    use_ignore=True)

        return object_dict
