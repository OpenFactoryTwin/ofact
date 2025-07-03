"""
# TODO Add Module Description
@last update: ?.?.2022
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING, Optional

# Imports Part 2: PIP Imports

# Imports Part 3: Project Imports
if TYPE_CHECKING:
    from ofact.twin.state_model.processes import ProcessExecution
    from ofact.twin.agent_control.behaviours.env_release.interface import EnvInterfaceBehaviour


class Action:
    class Types(Enum):
        NO_ACTION = "NO_ACTION"
        PROCESS_EXECUTION = "PROCESS_EXECUTION"

    def __init__(self, type_: Types, agent_name: str):
        self.type_ = type_
        self.agent_name = agent_name


class ProcessExecutionAction(Action):

    def __init__(self, type_: Action.Types.PROCESS_EXECUTION, agent_name: str,
                 planned_process_execution: ProcessExecution, env_interface_behaviour: EnvInterfaceBehaviour,
                 ape_subscription: bool = False, deviation_tolerance_time_delta: Optional[timedelta] = None,
                 notification_time_delta: Optional[timedelta] = None):
        super(ProcessExecutionAction, self).__init__(type_=type_, agent_name=agent_name)
        self.planned_process_execution = planned_process_execution
        self.env_interface_behaviour = env_interface_behaviour
        self.ape_subscription = ape_subscription
        if deviation_tolerance_time_delta is None:
            raise Exception
        self.deviation_tolerance_time_delta = deviation_tolerance_time_delta
        self.notification_time_delta = notification_time_delta
