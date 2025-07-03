"""
Entry point to simulation for the bicycle world.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union, Optional, Dict, TYPE_CHECKING

from ofact.twin.repository_services.persistence import serialize_state_model
from ofact.twin.agent_control.organization import ThreePhasedAgents
from ofact.twin.agent_control.simulate import SimulationStarter
from ofact.env.simulation.event_discrete import EventDiscreteSimulation
from ofact.planning_services.model_generation.resource_settings import (get_resource_performance,
                                                                        update_resource_performances,
                                                                        get_schedules_resource,
                                                                        update_digital_twin_schedules)
from projects.bicycle_world.twin.agent_control.agents import MappingAgents

# from projects.bicycle_world.models.skill_matrix.reader import update_worker_abilities

if TYPE_CHECKING:
    from ofact.env.interfaces import ProgressTrackerSimulation
    from datetime import datetime

    from ofact.twin.state_model.model import StateModel


class SimulationStarterBicycleWorld(SimulationStarter):
    mapping_class_agents = MappingAgents
    environment_class = EventDiscreteSimulation
    agents_organization = ThreePhasedAgents

    def simulate(self,
                 digital_twin: StateModel,
                 agents_file_name: Optional[str] = None,
                 digital_twin_update_paths: Optional[Dict[str, str]] = None,
                 progress_tracker: Optional[ProgressTrackerSimulation] = None,
                 order_agent_amount: int = None,
                 start_time_simulation: Optional[datetime] = None,
                 simulation_end: Tuple[ThreePhasedAgents.SimulationEndTypes, Union[str, int], int] = (
                         ThreePhasedAgents.SimulationEndTypes.ORDER, [300]),
                 order_target_quantity: int = 19,
                 digital_twin_result_path=None):
        state_model = super().simulate(digital_twin=digital_twin,
                                       agents_file_name=agents_file_name,
                                       digital_twin_update_paths=digital_twin_update_paths,
                                       progress_tracker=progress_tracker, order_agent_amount=order_agent_amount,
                                       start_time_simulation=start_time_simulation, simulation_end=simulation_end,
                                       order_target_quantity=order_target_quantity,
                                       digital_twin_result_path=digital_twin_result_path)
        return state_model

    def _update_digital_twin(self, digital_twin, digital_twin_update_paths, start_time_simulation):
        if not digital_twin_update_paths:
            return digital_twin

        resource_schedule_file_name = digital_twin_update_paths["resource_schedule"]
        resource_schedule_path = Path(self.project_path +
                                      f'{self.path_to_models}/models/resource/{resource_schedule_file_name}')
        resources = digital_twin.get_all_resources()
        resource_available_times = get_schedules_resource(resource_schedule_path)
        update_digital_twin_schedules(resources, resource_available_times,
                                      start_time_stamp=start_time_simulation)
        # resource_performances = get_resource_performance(resource_schedule_path,
        #                                                  sheet_name="General")
        # update_resource_performances(resources, resource_performances)

        return digital_twin

    def _store_results(self, digital_twin, digital_twin_result_path):
        serialize_state_model(state_model=digital_twin, target_file_path=digital_twin_result_path,
                              dynamics=True)
        print("Stored digital twin")
