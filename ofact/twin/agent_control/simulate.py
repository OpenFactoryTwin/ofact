"""
Entry point to simulate
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union, Optional, Dict, Coroutine

import spade

from ofact.planning_services.model_generation.persistence import serialize_state_model
from ofact.planning_services.model_generation.resource_settings import get_schedules_resource, \
    update_digital_twin_schedules, get_resource_performance
from ofact.twin.agent_control.administration import start_agent_control, stop_agent_control
from ofact.twin.agent_control.organization import Agents
from ofact.twin.change_handler.change_handler import ChangeHandlerSimulation
from ofact.env.simulation.event_discrete import EventDiscreteSimulation
from ofact.twin.repository_services.deserialization.agents_importer import MapperMethods, AgentsObjects, MappingAgents

if TYPE_CHECKING:
    from ofact.env.interfaces import ProgressTrackerSimulation
    from ofact.twin.state_model.model import StateModel

_container_instance = None


def get_container_instance():
    global _container_instance
    if _container_instance is None:
        _container_instance = spade.container.Container()
    else:
        _container_instance.reset()
    return _container_instance


def run_agent_simulation(main_func: Coroutine) -> None:  # pragma: no cover
    container = get_container_instance()
    try:
        res = {}
        async def wrapper():
            # await the real work…
            res["state_model"] = await main_func

        container.run(wrapper())
        return res.get("state_model")
    except KeyboardInterrupt:
        spade.logger.warning("Keyboard interrupt received. Stopping SPADE...")
    except Exception as e:  # pragma: no cover
        spade.logger.error("Exception in the event loop: {}".format(e))

    # container = get_container_instance()
    # server = None
    # try:
    #     spade.container.loguru.logger.remove()  # Silent server
    #     server_instance = spade.container.Server(
    #         spade.container.Parameters(host="localhost", database_in_memory=True)
    #     )
    #     server = container.loop.create_task(server_instance.start())
    #     container.run(server_instance.ready.wait())
    #     spade.container.logger.info("SPADE XMPP server running on localhost:5222")
    #
    #     container.run(main_func)
    #     container.stop_agents()
    #     return
    # except KeyboardInterrupt:
    #     spade.logger.warning("Keyboard interrupt received. Stopping SPADE...")
    # except Exception as e:  # pragma: no cover
    #     spade.logger.error("Exception in the event loop: {}".format(e))
    #
    # container.stop_agents()
    # if server:
    #     server.cancel()

    if sys.version_info >= (3, 7):  # pragma: no cover
        tasks = asyncio.all_tasks(loop=container.loop)  # pragma: no cover
    else:
        tasks = asyncio.Task.all_tasks(loop=container.loop)  # pragma: no cover
    for task in tasks:
        task.cancel()
        with spade.container.suppress(asyncio.CancelledError):
            container.run(task)

    container.loop.run_until_complete(container.loop.shutdown_asyncgens())

    container.loop.close()
    print("Before reset...")
    container.__agents = {}
    container.is_running = False
    spade.logger.debug("Loop closed")

    return main_func


class SimulationStarter:
    mapping_class_agents = MappingAgents
    environment_class = EventDiscreteSimulation
    agents_organization = Agents

    def __init__(self, xmpp_server_ip_address, xmpp_server_rest_api_port, xmpp_server_shared_secret,
                 xmpp_server_rest_api_users_endpoint, project_path, path_to_models="",
                 order_agent_amount_digital_twin_dependent: bool = False):
        self.mode = "SIMULATION"
        self.host = SimulationStarter.get_host(xmpp_server_ip_address, xmpp_server_rest_api_port)
        self.xmpp_server_ip_address = xmpp_server_ip_address
        self.xmpp_server_shared_secret = xmpp_server_shared_secret
        self.xmpp_server_rest_api_users_endpoint = xmpp_server_rest_api_users_endpoint

        self.project_path = str(project_path)
        self.path_to_models = path_to_models

        self.order_agent_amount_digital_twin_dependent = order_agent_amount_digital_twin_dependent

    @classmethod
    def get_host(cls, xmpp_server_ip_address, xmpp_server_rest_api_port):
        return "http://" + xmpp_server_ip_address + xmpp_server_rest_api_port

    def simulate(self,
                 digital_twin: StateModel,
                 agents_file_name: Optional[str] = None,
                 digital_twin_update_paths: Optional[Dict[str, Optional[str]]] = None,
                 progress_tracker: Optional[ProgressTrackerSimulation] = None,
                 order_agent_amount: int = None,
                 start_time_simulation: Optional[datetime] = None,
                 simulation_end: Tuple[Agents.SimulationEndTypes, Union[str, int], int] = (
                         Agents.SimulationEndTypes.ORDER, [300]),
                 order_target_quantity: int = 40,
                 digital_twin_result_path: Optional[Union[str, Path]] = None):
        """
        Execute the simulation
        :digital_twin_objects:
        ToDo: comment
        :simulation_end_definition:
        - ORDER: last order is completed/ has reached a predefined value (values: int | COMPLETE)
            -> realized in the order_pool agent
        - TIME_LIMIT: time limit has been reached (values: datetime)
        """
        state_model = run_agent_simulation(
            self._execute_simulation(
                digital_twin,
                agents_file_name,
                digital_twin_update_paths,
                progress_tracker,
                order_agent_amount,
                start_time_simulation,
                simulation_end,
                order_target_quantity,
                digital_twin_result_path))
        return state_model

    async def _execute_simulation(self,
                                  state_model: StateModel,
                                  agents_file_name: Optional[str] = None,
                                  digital_twin_update_paths: Optional[Dict[str, Optional[str]]] = None,
                                  progress_tracker: Optional[ProgressTrackerSimulation] = None,
                                  order_agent_amount: int = None,
                                  start_time_simulation: Optional[datetime] = None,
                                  simulation_end: Tuple[Agents.SimulationEndTypes, Union[str, int], int] = (
                                          Agents.SimulationEndTypes.ORDER, [300]),
                                  order_target_quantity: int = 40,
                                  digital_twin_result_path: Optional[Union[str, Path]] = None):
        """
        Execute the simulation.
        Including: Build the environment, execute the actual simulation, store the simulation results.
        """
        start_time = datetime.now()
        print(start_time, "Execute Simulation")

        environment, change_handler, agents_model, agents_objects, scenario_name = (
            self._set_up_simulation_environment(order_target_quantity, order_agent_amount, digital_twin_update_paths,
                                                agents_file_name, progress_tracker, start_time_simulation, state_model,
                                                simulation_end))
        environment_established_time = datetime.now()
        print(environment_established_time, "Register Agents")

        all_agents = await self._register_agents(environment, agents_model, instance=scenario_name)

        agents_registered_time = datetime.now()
        self._start_simulation(change_handler)
        await self.run_simulation(agents_model)
        _, end_time_simulation = state_model.get_consideration_period()
        simulation_ended_time = datetime.now()

        if digital_twin_result_path is not None:
            print("Store results")
            self._store_results(state_model, digital_twin_result_path)
        simulation_results_stored_time = datetime.now()

        self._stop_simulation(all_agents, agents_objects)
        simulation_environment_down_time = datetime.now()

        print(simulation_environment_down_time, "Simulation ended")

        # time analytics
        print(f"""
Environment Set Up Time:  {(datetime.min + (environment_established_time - start_time)).strftime("%H:%M:%S")}
Agent Registration Time:  {(datetime.min + (agents_registered_time - environment_established_time)).strftime("%H:%M:%S")}
Simulation Time:          {(datetime.min + (simulation_ended_time - agents_registered_time)).strftime("%H:%M:%S")}
Result Storing Time:      {(datetime.min + (simulation_results_stored_time - simulation_ended_time)).strftime("%H:%M:%S")}
Simulation shutdown Time: {(datetime.min + (simulation_environment_down_time - simulation_results_stored_time)).strftime("%H:%M:%S")}
==========================
Total Time: {simulation_environment_down_time - start_time}
Simulated Time: {(datetime.min + (end_time_simulation - start_time_simulation)).strftime("%H:%M:%S")}
Start: {start_time_simulation:%Y-%m-%d %H:%M:%S}
End  : {end_time_simulation :%Y-%m-%d %H:%M:%S}
""")
        return state_model
        # raise BaseException("Stopping execution")
        # exit()
        # return digital_twin

    def _set_up_simulation_environment(self, order_target_quantity, order_agent_amount, digital_twin_update_paths,
                                       agents_file_name, progress_tracker, start_time_simulation, digital_twin,
                                       simulation_end):

        simulation_end = (simulation_end[0], tuple(list(simulation_end[1]) + [order_target_quantity]))
        # logging.basicConfig(filename='agent_control.log', level=logging.DEBUG)

        digital_twin = self._update_digital_twin(digital_twin, digital_twin_update_paths, start_time_simulation)
        print("Digital Twin updated")
        if agents_file_name is None:
            raise Exception
        scenario_name = agents_file_name.split(".")[0]

        agents_xlsx_path = Path(self.project_path + f'{self.path_to_models}/models/agents/{agents_file_name}')

        empty_agents_model = self._get_empty_agents_model(progress_tracker, simulation_end)
        environment, change_handler = self._initialize_environment(start_time_simulation, digital_twin,
                                                                   empty_agents_model)
        print("Environment Initialized")
        agents_model, agents_objects = self._get_agents_model(digital_twin, empty_agents_model,
                                                              change_handler, order_agent_amount, order_target_quantity,
                                                              agents_xlsx_path, start_time_simulation)
        print("Agents Model Initialized")
        self._set_processes_to_stop(environment, agents_model)

        return environment, change_handler, agents_model, agents_objects, scenario_name

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

        return digital_twin

    def _get_empty_agents_model(self, progress_tracker, simulation_end):
        agents_organization = type(self).agents_organization
        empty_agents_model = agents_organization(progress_tracker=progress_tracker,
                                                 simulation_end=simulation_end)
        return empty_agents_model

    def _initialize_environment(self, start_time_simulation, digital_twin, empty_agents_model):
        if start_time_simulation is None:
            start_time_simulation = datetime.now()

        processes_to_stop = set()  # defined by agents?? - should be the loading processes and the value added processes
        environment_class = type(self).environment_class
        environment = environment_class(change_handler=None,
                                        start_time=start_time_simulation,
                                        processes_to_stop=processes_to_stop)

        change_handler = ChangeHandlerSimulation(digital_twin=digital_twin,
                                                 environment=environment,
                                                 agents=empty_agents_model)
        environment.add_change_handler(change_handler)

        return environment, change_handler

    def _get_agents_model(self, digital_twin, empty_agents_model, change_handler,
                          order_agent_amount, order_target_quantity, agents_xlsx_path, start_time_simulation):

        MapperMethods(digital_twin=digital_twin,
                      agents=empty_agents_model,
                      change_handler=change_handler)
        order_agent_amount = self._get_order_agent_amount(digital_twin, order_agent_amount, order_target_quantity,
                                                          start_time_simulation)

        agents_objects = AgentsObjects(path=agents_xlsx_path,
                                       digital_twin=digital_twin,
                                       empty_agents_model=empty_agents_model,
                                       mapping_class=type(self).mapping_class_agents,
                                       order_agent_amount=order_agent_amount)

        # ToDo: performance - number_of_orders_in_progress
        agents_model = agents_objects.get_agents()

        return agents_model, agents_objects

    def _get_order_agent_amount(self, digital_twin, order_agent_amount, order_target_quantity, start_time_simulation):
        if self.order_agent_amount_digital_twin_dependent:
            order_agent_amount = digital_twin.get_number_of_orders_in_progress(start_time_simulation)

            if order_agent_amount < order_target_quantity:
                order_agent_amount = order_target_quantity

        return order_agent_amount

    def _individualize_agents(self, environment, agents_model, instance):
        """Allows to have different simulations at the same time - to avoid the same name for agents"""
        print("Individualize Agents")
        for agent_list in list(agents_model.agents.values()):
            for agent in agent_list:
                agent.add_instance_suffix(instance)

        agent_names = [agent.name.lower()
                       for agent_list in list(agents_model.agents.values())
                       for agent in agent_list]
        agents_model.update_initial(environment)

        return agent_names

    def _set_processes_to_stop(self, environment, agents_model):
        order_agent = [agents
                       for agent_type, agents in agents_model.agents.items()
                       if "OrderDigitalTwin" in agent_type.__name__][0][0]
        processes_to_stop = set(order_agent.processes["loading_processes"])
        environment.processes_to_stop = processes_to_stop

    async def _register_agents(self, environment, agents_model, instance):
        """Register the agents to the xmpp server"""
        agent_names = self._individualize_agents(environment, agents_model, instance=instance)

        all_agents = \
            await start_agent_control(mode=self.mode,
                                      host=self.host,
                                      secret=self.xmpp_server_shared_secret,
                                      endpoint=self.xmpp_server_rest_api_users_endpoint,
                                      agent_names=agent_names,
                                      agents_model=agents_model)

        return all_agents

    def _start_simulation(self, change_handler):
        change_handler.start_simulation()

    async def run_simulation(self, agents_model):
        print("[main] Running until user interrupts with ctrl+C")
        order_pool_agents = [agents
                             for agent_type, agents in agents_model.agents.items()
                             if "OrderPoolDigitalTwinAgent" in agent_type.__name__]
        if order_pool_agents:
            await spade.wait_until_finished(order_pool_agents[0][0])

    def _store_results(self, digital_twin, digital_twin_result_path):
        serialize_state_model(state_model=digital_twin, target_file_path=digital_twin_result_path,
                              dynamics=True)
        print("Stored digital twin")

    def _stop_simulation(self, all_agents, agents_objects):

        stop_agent_control(all_agents)

        del agents_objects  # singleton object with weak references (allow to start the simulation a second time ...)
