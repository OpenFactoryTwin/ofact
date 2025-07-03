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

Encapsulates all kins of distributions, that return a random number. All Distribution has tbe
derived from ProbabilityDistribution and implement get_random_number()
classes:
    ProbabilityDistribution
    SingleValueDistribution
    BernoulliDistribution
    NormalDistribution

@contact persons: Christian Schwede & Adrian Freiter
@last update: 14.05.2024
"""

# Imports Part 1: Standard Imports
from abc import ABCMeta, abstractmethod
from copy import copy
from random import random

# Imports Part 2: PIP Imports
import numpy as np

# Imports Part 3: Project Imports
from ofact.twin.state_model.serialization import Serializable


class ProbabilityDistribution(Serializable, metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def get_random_number(self,
                          weight=None,
                          negative_values_allowed: bool = False):
        pass

    def get_expected_value(self):
        pass

    @classmethod
    @abstractmethod
    def get_parameters(cls) -> dict[str, object]:
        parameters: dict
        return parameters

    @abstractmethod
    def get_parameters_with_values(self) -> dict[str, object]:
        parameters: dict
        return parameters

    @abstractmethod
    def copy(self):
        """Copy the object with the same identification."""
        probability_distribution_copy = copy(self)

        return probability_distribution_copy

    def representation(self):
        """The representation of the object is unambiguous"""
        items = ("%s = %r" % (k, v)
                 for k, v in self.__dict__.items())
        object_representation = "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))
        return object_representation


class SingleValueDistribution(ProbabilityDistribution):

    def __init__(self,
                 value: float):
        super().__init__()
        if not isinstance(value, float):
            try:
                value = float(value)
            except ValueError:
                raise Exception(f"The value '{value}' is not of type float")
        self.value: float = value

    def copy(self):
        """Copy the object with the same identification."""
        single_value_distribution_copy = super(SingleValueDistribution, self).copy()

        return single_value_distribution_copy

    def get_random_number(self, weight=None,
                          negative_values_allowed: bool = False):
        if negative_values_allowed:
            return self.value
        elif self.value > 0.0:
            return self.value
        else:
            return 0

    def get_expected_value(self):
        return self.value

    @classmethod
    def get_parameters(cls) -> dict[str, object]:
        parameters = {"value": float}
        return parameters

    def get_parameters_with_values(self) -> dict[str, object]:
        parameters = {"value": self.value}
        return parameters


class BernoulliDistribution(ProbabilityDistribution):

    def __init__(self,
                 probability: float,
                 not_successful_value: float = 0.0,
                 successful_value: float = 1.0):
        super().__init__()
        self.probability: float = probability  # probability for a not successful value
        self.not_successful_value: float = not_successful_value
        self.successful_value: float = successful_value

    def copy(self):
        """Copy the object with the same identification."""
        bernoulli_distribution_copy = super(BernoulliDistribution, self).copy()

        return bernoulli_distribution_copy

    def get_random_number(self, weight=None, negative_values_allowed: bool = False):
        sample = random()
        if weight is not None:
            sample *= weight
        if random() <= self.probability:
            random_value = self.successful_value
        else:
            random_value = self.not_successful_value

        if negative_values_allowed:
            return random_value
        elif random_value > 0.0:
            return random_value
        else:
            return 0

    @classmethod
    def get_parameters(cls) -> dict[str, object]:
        parameters = {"probability": float,
                      "not_successful_value": float,
                      "successful_value": float}
        return parameters

    def get_parameters_with_values(self) -> dict[str, object]:
        parameters = {"probability": self.probability,
                      "not_successful_value": self.not_successful_value,
                      "successful_value": self.successful_value}
        return parameters


class NormalDistribution(ProbabilityDistribution):

    def __init__(self,
                 mue: float,
                 sigma: float = 1):
        super().__init__()
        self.mue = mue

        self.sigma: float = sigma

    def copy(self):
        """Copy the object with the same identification."""
        normal_distribution_copy = super(NormalDistribution, self).copy()

        return normal_distribution_copy

    def get_random_number(self, weight=None, negative_values_allowed: bool = False):
        # ToDo: Christian - a general way to set frontiers

        if self.sigma is None:
            self.sigma = 0
        random_value = np.random.normal(loc=self.mue, scale=self.sigma)

        if negative_values_allowed:
            return random_value
        elif random_value > 0.0:
            return random_value
        else:
            return 0

    def get_expected_value(self):
        return self.mue

    @classmethod
    def get_parameters(cls) -> dict[str, object]:
        parameters = {"mue": float,
                      "sigma": float}
        return parameters

    def get_parameters_with_values(self) -> dict[str, object]:
        parameters = {"mue": self.mue,
                      "sigma": self.sigma}
        return parameters
