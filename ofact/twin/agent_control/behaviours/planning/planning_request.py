"""Build a standard for the planning requests"""

# Imports Part 1: Standard imports
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour

if TYPE_CHECKING:
    from ofact.twin.agent_control.behaviours.negotiation.objects import (
        ProcessCallForProposal, PartCallForProposal, ResourceCallForProposal, ProcessGroupCallForProposal)


class PlanningRequest(DigitalTwinCyclicBehaviour, metaclass=ABCMeta):
    """
    The planning request is triggered by the process_request method that
    should be overwritten by the respective request types ...
    """

    @abstractmethod
    async def process_request(self, negotiation_object_request: ProcessCallForProposal | PartCallForProposal |
                                                                ResourceCallForProposal | ProcessGroupCallForProposal) \
            -> (bool, list, list):
        pass
