from __future__ import annotations

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Dict
import random
import math
from datetime import datetime, timedelta
import time
from pathlib import Path
from ofact.twin.state_model.model import ProcessExecutionTypes

import pandas as pd

from ofact.planning_services.scenario_generation.state_model_adaption import StateModelAdaption
from ofact.planning_services.scenario_generation.agent_model_adaption import AgentModelAdaption
from ofact.planning_services.model_generation.persistence import serialize_state_model

if TYPE_CHECKING:
    from ofact.twin.agent_control.simulate import SimulationStarter
    from ofact.twin.state_model.model import StateModel

class Optimizer(metaclass=ABCMeta):

    def __init__(self, state_model: StateModel, agents_model_file, agents_model_file_path,
                 simulation_starter: SimulationStarter, simulation_start_time: datetime, scenario_settings: Dict,
                 project_path: Path, result_path: Path):

        self._simulation_starter = simulation_starter
        self._simulation_start_time = simulation_start_time

        self._project_path = project_path
        self._result_path = result_path

        self._state_model = state_model
        self._agents_model_file_path = agents_model_file_path
        self._agents_model_file = agents_model_file

        self.order_agent_amount = 12

        self._scenario_settings = scenario_settings

        self._state_model_adaption = StateModelAdaption(self._state_model)
        agent_model_df = pd.read_excel(self._agents_model_file_path)
        self._agent_model_adaption = AgentModelAdaption(self._state_model, agent_model_df)

    @abstractmethod
    def optimize(self):
        pass

    def _simulate(self, state_model):
        print("\nExecuting Simulation")
        self._simulation_starter.simulate(
            digital_twin=state_model,
            start_time_simulation=datetime.now(),
            digital_twin_update_paths=self._scenario_settings,
            agents_file_name=self._agents_model_file,
            order_agent_amount=self.order_agent_amount,
            # digital_twin_result_path=self._result_path  # ToDo: Maybe only required for the end results
        )

        # if not self._result_path.exists():
        #     raise FileNotFoundError(f"Simulation result file not created: {self._result_path}")

        simulated_model = state_model
        return simulated_model


class GeneticAlgorithmOptimizer(Optimizer, metaclass=ABCMeta):

    def __init__(self, state_model: StateModel, agents_model_file, agents_model_file_path,
                 simulation_starter: SimulationStarter,
                 simulation_start_time: datetime, scenario_settings: Dict,
                 project_path: Path, result_path: Path,
                 num_generations=10, population_size=6, mutation_rate=0.2):
        super().__init__(state_model=state_model, agents_model_file=agents_model_file,
                         agents_model_file_path=agents_model_file_path, simulation_starter=simulation_starter,
                         simulation_start_time=simulation_start_time, scenario_settings=scenario_settings,
                         project_path=project_path, result_path=result_path)

        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def optimize(self):
        population = [self._generate_initial_scenario()
                      for _ in range(self.population_size)]
        print("Initial population generated:")
        print(population)

        for generation in range(self.num_generations):
            print(f"\n🔁 Evaluating generation {generation + 1}/{self.num_generations}")
            start_gen_time = time.time()

            scored = []
            best_score = -9999
            for i, chrom in enumerate(population):
                print(f"   ➤ Individual {i + 1}/{len(population)} in generation {generation + 1}...")

                individual_start_time = time.time()
                simulated_state_model, score = self._get_trial_evaluation(chrom)
                individual_time = time.time() - individual_start_time

                print(f"      ⏱ Done in {individual_time:.2f} seconds")
                if score > best_score:
                    best_score = score
                    # save the state model with the best schedule
                    serialize_state_model(simulated_state_model, target_file_path=self._result_path, dynamics=True)

                if score != -9999:
                    scored.append((chrom, score))

            if not scored:
                print("\n⚠️ No valid evaluations found in this generation.")
                return None

            scored.sort(key=lambda x: x[1], reverse=True)
            print(f"✅ Best Score in Generation {generation + 1}: {scored[0][1]:.4f}")

            gen_time = time.time() - start_gen_time
            time_remaining = gen_time * (self.num_generations - generation - 1)

            print(f"⏳ Generation Time: {timedelta(seconds=gen_time)}")
            print(f"⌛ Estimated Time Left: {timedelta(seconds=int(time_remaining))}")

            # --- Crossover + Mutation ---
            parents = [chrom for chrom, _ in scored[:max(2, self.population_size // 2)]]

            # 🛑 Exit if not enough parents to proceed
            if len(parents) < 2:
                print("\n⚠️ Not enough valid parents to continue. Optimization stopped.")
                return scored[0][0] if scored else None

            next_population = []

            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self._crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                next_population.append(child)

            if score != -9999:
                    scored.append((chrom, score))

            if not scored:
                print("\nNo valid evaluations found in this generation.")
                return None

            scored.sort(key=lambda x: x[1], reverse=True)
            print(f"\n Generation {generation + 1}: Best Score = {scored[0][1]:.4f}")

            parents = [chrom for chrom, _ in scored[:self.population_size // 2]]
            print(f" Valid individuals this generation: {len(scored)}")
            print(f" Parents available: {len(parents)}")

            if len(parents) < 2:
                print(" Not enough valid parents to proceed with crossover. Ending optimization early.")
                return scored[0][0] if scored else None

            next_population = []

            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self._crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                next_population.append(child)

            population = next_population

        best_solution = scored[0][0] if scored else None
        print("\n Optimization complete.")

        return best_solution

    @abstractmethod
    def _generate_initial_scenario(self) -> Dict:
        pass

    @abstractmethod
    def _get_trial_evaluation(self, scenario_input_parameters: Dict) -> [StateModel, float]:
        pass

    @abstractmethod
    def _prepare_scenario(self, scenario_input_parameters):
        pass

    @abstractmethod
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        pass

    @abstractmethod
    def _mutate(self, individual: Dict) -> Dict:
        pass


class ResourceAvailabilityGeneticAlgorithmOptimizer(GeneticAlgorithmOptimizer):

    def __init__(self, state_model: StateModel, agents_model_file, agents_model_file_path,
                 simulation_starter: SimulationStarter,
                 simulation_start_time: datetime, scenario_settings: Dict,
                 project_path: Path, result_path: Path,
                 num_generations=30, population_size=12, mutation_rate=0.2):
        super().__init__(state_model=state_model, agents_model_file=agents_model_file,
                         agents_model_file_path=agents_model_file_path, simulation_starter=simulation_starter,
                         simulation_start_time=simulation_start_time, scenario_settings=scenario_settings,
                         project_path=project_path, result_path=result_path,
                         num_generations=num_generations, population_size=population_size,
                         mutation_rate=mutation_rate)

        self._schedule_df = pd.read_excel(project_path + "/scenarios/optimization/models/resource/schedule_s1.xlsx",
            sheet_name="General"
        ).set_index("MA")

        self._work_stations = self._state_model.get_work_stations()
        self._main_part_agvs = [resource
                                for resource in self._state_model.get_active_moving_resources()
                                if "Main Part AGV" in resource.name]
        self.resources = self._work_stations + self._main_part_agvs
        self.resource_names = [resource.name
                               for resource in self.resources]

        self.required_order_lead_time_min = self._state_model.get_estimated_order_lead_time_mean()

        self.num_time_slots = 38

    def _generate_initial_scenario(self):
        return {
            "resources": self._work_stations + self._main_part_agvs,
            "availability": {
                res.name: list(self._schedule_df.loc[res.name].values[:self.num_time_slots])
                if res.name in self._schedule_df.index
                else [random.randint(0, 1) for _ in range(self.num_time_slots)]
                for res in self._work_stations + self._main_part_agvs
            }
        }

    def _get_trial_evaluation(self, scenario_input_parameters: Dict) -> [StateModel, float]:
        self._prepare_scenario(scenario_input_parameters)
        state_model = deepcopy(self._state_model)
        simulated_model = self._simulate(state_model)

        try:
            #  Only use resources that are actually scheduled (have at least one 1)
            active_resource_names = [name
                                     for name, schedule in scenario_input_parameters["availability"].items()
                                     if any(schedule)]

            rc_util = simulated_model.get_resource_capacity_utilization(active_resource_names)
            order_lead_time_mean = simulated_model.get_order_lead_time_mean()
            reliability = simulated_model.get_delivery_reliability()
        except Exception as e:
            print(f"Error during KPI extraction: {e}")
            rc_util, order_lead_time_mean, reliability = None, None, None

        if any(metric is None for metric in [rc_util, order_lead_time_mean, reliability]):
            print("KPI extraction failed. Skipping this chromosome.")
            return simulated_model, -9999

        #  Print KPI values for each individual
        if rc_util and len(rc_util) > 0:
            avg_util = sum(rc_util) / len(rc_util)
            print(f"  ▶ Resource Utilization (avg): {avg_util:.4f}")
        else:
            print("  ▶ Resource Utilization (avg): No data (empty list)")

        print(f"  ▶ Order Lead Time (s): {order_lead_time_mean}")
        if isinstance(reliability, (int, float)):
            print(f"  ▶ Delivery Reliability: {reliability:.2f}")
        else:
            print(f"  ▶ Delivery Reliability: {reliability}")

        evaluation = self._calculate_evaluation(rc_util, order_lead_time_mean, reliability)

        return simulated_model, evaluation

    def _prepare_scenario(self, scenario_input_parameters):
        schedule_start_time = datetime(2024, 10, 22, 6, 45)
        slot_duration = timedelta(minutes=15)
        all_resources = self._state_model.get_all_resources()

        for res_name, availability in scenario_input_parameters["availability"].items():
            resource_obj = next((r for r in all_resources if getattr(r, "name", "") == res_name), None)
            if not resource_obj:
                print(f" Resource {res_name} not found!")
                continue

            print(f"Resource found: {resource_obj.name}")

            try:
                process_executions = self._state_model.get_process_executions_list_for_resource(
                    ProcessExecutionTypes.ACTUAL,
                    resource_obj
                )
            except Exception as e:
                print(f"Error while getting executions: {e}")
                continue

            if not isinstance(process_executions, list) or len(process_executions) == 0:
                print(f" No process executions found for resource {res_name}, skipping block_period calls.")
                continue

            pe_id = process_executions[0].identification
            wo_id = process_executions[0].order.identification

            for i, available in enumerate(availability):
                if available == 0:
                    start = schedule_start_time + i * slot_duration
                    end = start + slot_duration
                    try:
                        if 'Surly Long Haul Trucker Rahmenkit' in str(process_executions[0].required_parts):
                            continue

                        resource_obj.block_period(
                            start,
                            end,
                            blocker_name="GA_OPT",
                            process_execution_id=pe_id,
                            work_order_id=wo_id
                        )
                    except Exception as e:
                        print(f" Failed to block time for {res_name} from {start} to {end}: {e}")

    def _calculate_evaluation(self, resource_capacity_utilization, order_lead_time_mean, delivery_reliability) -> float:
        # Handle utilization format
        if isinstance(resource_capacity_utilization, dict):
            values = list(resource_capacity_utilization.values())
            rc_util = sum(values) / len(values) if values else 0.0
        elif isinstance(resource_capacity_utilization, list):
            rc_util = sum(resource_capacity_utilization) / len(
                resource_capacity_utilization) if resource_capacity_utilization else 0.0
        else:
            rc_util = float(resource_capacity_utilization)

        #  Convert order_lead_time_mean to minutes
        if isinstance(order_lead_time_mean, timedelta):
            order_lead_time_mean = order_lead_time_mean.total_seconds() / 60  # convert to minutes

        # Fail-safe for missing KPIs
        if order_lead_time_mean is None:
            order_lead_time_mean = 999.0
        if delivery_reliability is None or math.isnan(delivery_reliability):
            delivery_reliability = 0.0

        # Weighted score calculation
        order_lead_time_mean_normalized = ((order_lead_time_mean - self.required_order_lead_time_min) /
                                           self.required_order_lead_time_min)
        score = (1.5 * rc_util) - (2.0 * order_lead_time_mean_normalized) + (3.0 * delivery_reliability)

        return score

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        child_availability = {}

        for res_name in parent1["availability"]:
            slots1 = parent1["availability"][res_name]
            slots2 = parent2["availability"][res_name]
            child_availability[res_name] = [random.choice([a, b]) for a, b in zip(slots1, slots2)]

        return {"availability": child_availability}

    def _mutate(self, individual: Dict) -> Dict:
        res_name = random.choice(self.resource_names)  #  use the resource name as the key
        idx = random.randint(0, self.num_time_slots - 1)

        if res_name in individual["availability"]:
            individual["availability"][res_name][idx] ^= 1

        return individual
