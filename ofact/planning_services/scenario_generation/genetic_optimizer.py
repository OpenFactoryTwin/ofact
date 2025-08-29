# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
import time
import random
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple
from abc import ABCMeta, abstractmethod


import pandas as pd

# ofact imports
from ofact.twin.state_model.model import ProcessExecutionTypes
from ofact.planning_services.scenario_generation.state_model_adaption import StateModelAdaption
from ofact.planning_services.scenario_generation.agent_model_adaption import AgentModelAdaption
from ofact.planning_services.model_generation.persistence import serialize_state_model, deserialize_state_model

if TYPE_CHECKING:
    from ofact.twin.agent_control.simulate import SimulationStarter
    from ofact.twin.state_model.model import StateModel


# -----------------------------
# Helpers used by worker process
# -----------------------------

def _apply_delivery_dates(state_model, delivery_dates: List[datetime], verbose: bool = False):
    orders = sorted(state_model.get_orders(), key=lambda o: o.identification)
    if not delivery_dates:
        return
    if len(orders) != len(delivery_dates) and verbose:
        print("Warning: Number of delivery dates does not match number of orders.")
    for order, new_delivery_date in zip(orders, delivery_dates):
        if verbose:
            print(f"Updating order {order.identification} delivery date from "
                  f"{order.delivery_date_planned} to {new_delivery_date}")
        order.delivery_date_planned = new_delivery_date


def _apply_availability_to_model(state_model, availability: Dict[str, List[int]], verbose: bool = False):
    """Block resource time windows for zeros in availability."""
    schedule_start_time = datetime(2025, 7, 4, 0, 0, 0)
    slot_duration = timedelta(minutes=15)
    all_resources = state_model.get_all_resources()

    # quick map for case-insensitive lookup
    name_map = {}
    for r in all_resources:
        n = getattr(r, "name", "")
        if n:
            name_map[n.lower().strip()] = r

    for res_name, slots in availability.items():
        res_name_norm = res_name.lower().strip()
        resource_obj = name_map.get(res_name_norm)
        if not resource_obj:
            if verbose:
                print(f" Resource '{res_name}' not found!")
            continue

        if verbose:
            print(f"Resource found: {resource_obj.name}")

        try:
            process_executions = state_model.get_process_executions_list_for_resource(
                ProcessExecutionTypes.ACTUAL, resource_obj
            )
        except Exception as e:
            if verbose:
                print(f"Error while getting executions for resource '{res_name}': {e}")
            continue

        if not process_executions:
            if verbose:
                print(f" No process executions found for resource '{res_name}', skipping block_period calls.")
            continue

        # Use first execution's ids for blocking context
        pe_id = process_executions[0].identification
        wo_id = process_executions[0].order.identification

        for i, available in enumerate(slots):
            if int(available) == 0:
                start = schedule_start_time + i * slot_duration
                end = start + slot_duration
                try:
                    resource_obj.block_period(
                        start,
                        end,
                        blocker_name="GA_OPT",
                        process_execution_id=pe_id,
                        work_order_id=wo_id
                    )
                except Exception as e:
                    if verbose:
                        print(f" Failed to block time for '{res_name}' from {start} to {end}: {e}")


def _proc_worker(
    agents_model_file: str,
    scenario_settings: dict,
    sim_config: dict,
    availability: dict,
    lead_min_minutes: float,
    lead_max_minutes: float,
    verbose: bool = False,
):
    """
    Runs one individual simulation in a separate process and returns:
        (score, rc_avg, lead_minutes, reliability)
    or ("ERR", traceback_string) on failure.

    IMPORTANT: We do NOT serialize the model here to avoid recursion errors.
    """
    try:
        import sys, io, contextlib, traceback
        sys.setrecursionlimit(max(5000, sys.getrecursionlimit()))

        # Load base model in THIS process
        state_model = deserialize_state_model(
            Path(sim_config["base_model_file_path"]),
            state_model_generation_settings=None,
            persistence_format="pkl",
            deserialization_required=False,
        )

        # Apply delivery date overrides + scenario availability
        dd = sim_config.get("delivery_dates", None)
        if dd:
            _apply_delivery_dates(state_model, dd, verbose=verbose)
        _apply_availability_to_model(state_model, availability, verbose=verbose)

        active_resource_names = [
            name for name, schedule in availability.items()
            if any(int(x) != 0 for x in schedule)
        ]

        port_raw = sim_config.get("xmpp_server_rest_api_port")
        port_str = str(port_raw).strip()
        if not port_str.startswith(":"):
            # ofact's get_host does "http://{ip}{port}" so keep the colon
            if port_str.isdigit() or ":" not in port_str:
                port_str = ":" + port_str

        starter = sim_config["simulation_starter"](
            project_path=Path(sim_config["project_path"]),
            path_to_models=sim_config["path_to_models"],
            xmpp_server_ip_address=sim_config["xmpp_server_ip_address"],
            xmpp_server_rest_api_port=port_str,
            xmpp_server_shared_secret=sim_config["xmpp_server_shared_secret"],
            xmpp_server_rest_api_users_endpoint=sim_config["xmpp_server_rest_api_users_endpoint"],
        )

        # Run simulation with suppressed stdout unless verbose=True
        if verbose:
            starter.simulate(
                digital_twin=state_model,
                start_time_simulation=datetime.now(),
                digital_twin_update_paths=scenario_settings,
                agents_file_name=agents_model_file,
                order_agent_amount=12,
            )
        else:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                starter.simulate(
                    digital_twin=state_model,
                    start_time_simulation=datetime.now(),
                    digital_twin_update_paths=scenario_settings,
                    agents_file_name=agents_model_file,
                    order_agent_amount=12,
                )

        # KPIs
        rc_util = state_model.get_resource_capacity_utilization(active_resource_names)
        lead = state_model.get_order_lead_time_mean()
        rel = state_model.get_delivery_reliability()

        # Normalize KPIs
        if isinstance(rc_util, dict):
            vals = list(rc_util.values())
            rc_avg = sum(vals) / len(vals) if vals else 0.0
        elif isinstance(rc_util, list):
            rc_avg = sum(rc_util) / len(rc_util) if rc_util else 0.0
        else:
            rc_avg = float(rc_util or 0.0)

        if hasattr(lead, "total_seconds"):
            lead_min = lead.total_seconds() / 60.0
        else:
            try:
                lead_min = float(lead)
            except Exception:
                lead_min = 999.0

        try:
            rel = float(rel)
        except Exception:
            rel = 0.0

        denom = max(lead_max_minutes - lead_min_minutes, 1e-9)
        ol_norm = (lead_min - lead_min_minutes) / denom
        # Optional clamp
        if math.isnan(ol_norm):
            ol_norm = 1.0
        else:
            ol_norm = max(0.0, min(1.0, ol_norm))

        score = 1.0 * rc_avg - 0.5 * ol_norm + 2.0 * rel

        return (float(score), float(rc_avg), float(lead_min), float(rel))

    except Exception:
        import traceback
        return ("ERR", traceback.format_exc())


# -----------------------------
# Base classes (unchanged API)
# -----------------------------

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

        from datetime import timedelta
        self.required_order_lead_time_min = timedelta(minutes=30)
        self.required_order_lead_time_max = timedelta(minutes=120)

        self._scenario_settings = scenario_settings

        self._state_model_adaption = StateModelAdaption(self._state_model)
        agent_model_df = pd.read_excel(self._agents_model_file_path)
        self._agent_model_adaption = AgentModelAdaption(self._state_model, agent_model_df)

    def optimize(self):
        raise NotImplementedError

    def _simulate(self, state_model):
        # NOTE: Do NOT rely on a result file being created; just run:
        print("\nExecuting Simulation")
        self._simulation_starter.simulate(
            digital_twin=state_model,
            start_time_simulation=datetime.now(),
            digital_twin_update_paths=self._scenario_settings,
            agents_file_name=self._agents_model_file,
            order_agent_amount=self.order_agent_amount,
        )
        # No result-path existence check (prevents false errors)
        return state_model


class GeneticAlgorithmOptimizer(Optimizer):

    def __init__(self, state_model: StateModel, agents_model_file, agents_model_file_path,
                 simulation_starter: SimulationStarter,
                 simulation_start_time: datetime, scenario_settings: Dict,
                 project_path: Path, result_path: Path,
                 num_generations=None, population_size=None, mutation_rate=None):
        super().__init__(state_model=state_model, agents_model_file=agents_model_file,
                         agents_model_file_path=agents_model_file_path, simulation_starter=simulation_starter,
                         simulation_start_time=simulation_start_time, scenario_settings=scenario_settings,
                         project_path=project_path, result_path=result_path)

        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def optimize(self):
        raise NotImplementedError

    def _generate_initial_scenario(self) -> Dict:
        raise NotImplementedError

    def _get_trial_evaluation(self, scenario_input_parameters: Dict) -> Tuple[StateModel, float]:
        raise NotImplementedError

    def _prepare_scenario(self, scenario_input_parameters):
        raise NotImplementedError

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        raise NotImplementedError

    def _mutate(self, individual: Dict) -> Dict:
        raise NotImplementedError


class ResourceAvailabilityGeneticAlgorithmOptimizer(GeneticAlgorithmOptimizer):

    def __init__(self, state_model: StateModel, agents_model_file, agents_model_file_path,
                 simulation_starter: SimulationStarter,
                 simulation_start_time: datetime, scenario_settings: Dict,
                 project_path: Path, result_path: Path,
                 initial_schedule_file_path: Path,
                 sim_config: dict,
                 num_generations=2, population_size=4, mutation_rate=0.5,
                 verbose: bool = False):

        super().__init__(state_model=state_model, agents_model_file=agents_model_file,
                         agents_model_file_path=agents_model_file_path, simulation_starter=simulation_starter,
                         simulation_start_time=simulation_start_time, scenario_settings=scenario_settings,
                         project_path=project_path, result_path=result_path,
                         num_generations=num_generations, population_size=population_size,
                         mutation_rate=mutation_rate)

        self.best_scores: List[float] = []
        self.mean_scores: List[float] = []
        self.worst_scores: List[float] = []

        self.global_best_individual = None
        self.global_best_score = float("-inf")

        # track GA best (for printing and external access)
        self.ga_best_score = None          # float
        self.ga_best_schedule = None       # dict[str, list[int]]
        self.ga_best_metrics = None        # tuple(rc_avg, lead_min, rel)


        self._verbose = verbose

        # Keep sim_config pickle-friendly (strings / lists / numbers)
        self._sim_config_sim_config = {
            "simulation_starter": self._simulation_starter,
            "project_path": str(sim_config["project_path"]),
            "path_to_models": str(sim_config["path_to_models"]),
            "xmpp_server_ip_address": str(sim_config["xmpp_server_ip_address"]),
            # keep as string; worker will ensure leading colon
            "xmpp_server_rest_api_port": str(sim_config["xmpp_server_rest_api_port"]),
            "xmpp_server_shared_secret": str(sim_config["xmpp_server_shared_secret"]),
            "xmpp_server_rest_api_users_endpoint": str(sim_config["xmpp_server_rest_api_users_endpoint"]),
            "base_model_file_path": str(sim_config["base_model_file_path"]),
            "delivery_dates": sim_config.get("delivery_dates"),
        }

        # Load initial schedule Excel (accept "Resource" or "MA" as name column)
        df = pd.read_excel(initial_schedule_file_path, sheet_name="General")
        name_col = "Resource" if "Resource" in df.columns else ("MA" if "MA" in df.columns else None)
        if name_col is None:
            raise ValueError("Initial schedule sheet must have 'Resource' or 'MA' column.")
        self._schedule_df = df.set_index(name_col)

        self._initial_resource_names = set(name.lower().strip() for name in self._schedule_df.index)

        all_work_stations = self._state_model.get_work_stations()
        all_main_part_agvs = [
            resource for resource in self._state_model.get_active_moving_resources()
            if "Main Part AGV" in resource.name
        ]

        self._work_stations = [r for r in all_work_stations if r.name.lower().strip() in self._initial_resource_names]
        self._main_part_agvs = [r for r in all_main_part_agvs if r.name.lower().strip() in self._initial_resource_names]

        self.resources = self._work_stations + self._main_part_agvs
        self.resource_names = [resource.name for resource in self.resources]

        # Use project's function for mean lead time if available
        self.required_order_lead_time_min = self._state_model.get_estimated_order_lead_time_mean()
        self.required_order_lead_time_max = timedelta(minutes=120)

        self.num_time_slots = len(self._schedule_df.columns)

    # ------- GA public API -------

    def optimize(self):
        population = [self._generate_initial_scenario() for _ in range(self.population_size)]
        print("Initial population generated:")
        print([{ "availability": p["availability"] } for p in population])

        from concurrent.futures import ProcessPoolExecutor, as_completed

        for generation in range(self.num_generations):
            print(f"\nEvaluating generation {generation + 1}/{self.num_generations}")

            # --- process-parallel evaluation ---
            scored: List[Tuple[Dict, float]] = []
            best_score = -9999.0
            best_availability = None

            lead_min_minutes = self.required_order_lead_time_min.total_seconds() / 60.0
            lead_max_minutes = self.required_order_lead_time_max.total_seconds() / 60.0

            jobs = []
            for idx, chrom in enumerate(population):
                if self._verbose:
                    print(f"   ➤ Prep Individual {idx + 1}/{len(population)}...")
                availability = chrom["availability"]
                jobs.append((idx, availability))

            start_parallel = time.time()
            with ProcessPoolExecutor(max_workers=min(self.population_size, os.cpu_count() or 1)) as ex:
                fut_to_idx = {}
                for (idx, availability) in jobs:
                    fut = ex.submit(
                        _proc_worker,
                        self._agents_model_file,
                        self._scenario_settings,
                        self._sim_config,
                        availability,
                        lead_min_minutes,
                        lead_max_minutes,
                        self._verbose,
                    )
                    fut_to_idx[fut] = (idx, availability)

                for fut in as_completed(fut_to_idx):
                    i, availability = fut_to_idx[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        print(f"  ✖ Individual {i + 1} failed (executor): {e}")
                        continue

                    if isinstance(result, tuple) and len(result) == 2 and result[0] == "ERR":
                        # Show last lines of worker traceback to keep logs short
                        lines = result[1].splitlines()
                        tail = "\n".join(lines[-8:])
                        print(f"  ✖ Individual {i + 1} worker error (last lines):\n{tail}")
                        continue

                    try:
                        score, rc_avg, lead_min, rel = result
                        print(
                            f"  ▶ Individual {i+1} | "
                            f"Score={score:.4f}, Util={rc_avg:.3f}, Lead={lead_min:.1f} min, Rel={rel:.3f}"
                        )
                        # Update GA-best (pre-validation) if this individual is better
                        if (self.ga_best_score is None) or (score > self.ga_best_score):
                            self.ga_best_score = score
                            # deep copy to avoid later mutations
                            self.ga_best_schedule = {k: list(v) for k, v in availability.items()}
                            self.ga_best_metrics = (rc_avg, lead_min, rel)

                    except Exception as e:
                        print(f"  ✖ Individual {i + 1} bad result format: {e}")
                        continue

                    if score > best_score:
                        best_score = score
                        best_availability = availability

                    # track global best across ALL generations
                    if score > self.global_best_score:
                        self.global_best_score = score
                        self.global_best_individual = {"availability": availability}
                        self.ga_best_score = score
                        self.ga_best_schedule = availability
                        self.ga_best_metrics = (rc_avg, lead_min, rel)

                    if score != -9999:
                        scored.append(({"availability": availability}, score))

            print(f"  ▶ Parallel eval wall time: {timedelta(seconds=(time.time() - start_parallel))}")

            if not scored:
                print("\n No valid evaluations found in this generation.")
                print("\n Optimization complete.")
                return None

            scored.sort(key=lambda x: x[1], reverse=True)

            best = scored[0][1]
            mean = sum(score for _, score in scored) / len(scored)
            worst = scored[-1][1]

            self.best_scores.append(best)
            self.mean_scores.append(mean)
            self.worst_scores.append(worst)

            print(f" Best Score in Generation {generation + 1}: {best:.4f}")

            # --- Crossover + Mutation ---
            parents = [chrom for chrom, _ in scored[:max(2, self.population_size // 2)]]
            if len(parents) < 2:
                print("\n Not enough valid parents to continue. Optimization stopped.")
                best_solution = parents[0] if parents else None
                print("\n Optimization complete.")
                return best_solution

            next_population = []
            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self._crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                next_population.append(child)

            population = next_population

        best_solution = {"availability": best_availability} if best_availability else None

        # ----- GA summary BEFORE any final validation run -----
        if self.ga_best_score is not None and self.ga_best_schedule is not None:
            print("\n================ GA SUMMARY (pre-validation) ================")
            print(f"Best KPI score found by GA: {self.ga_best_score:.4f}")

            if self.ga_best_metrics:
                rc_avg, lead_min, rel = self.ga_best_metrics
                print(f"  Util={rc_avg:.3f}, Lead={lead_min:.1f} min, Rel={rel:.3f}")

            print("\nBest availability schedule:")
            for res_name in sorted(self.ga_best_schedule.keys(), key=str.lower):
                print(f"{res_name}: {self.ga_best_schedule[res_name]}")
            print("=============================================================\n")

            # Save to Desktop (Windows-friendly)
            from datetime import datetime
            desktop = Path(os.path.expanduser("~")) / "Desktop"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            desktop_out = desktop / f"ga_best_schedule_pre_validation_{ts}.xlsx"
            self._export_schedule_excel(self.ga_best_schedule, desktop_out)

            # Also save to your project's results folder (optional)
            try:
                project_results_out = Path(self._result_path) / f"ga_best_schedule_pre_validation_{ts}.xlsx"
                self._export_schedule_excel(self.ga_best_schedule, project_results_out)
            except Exception:
                pass

        print("\n Optimization complete.")
        return best_solution

    def _export_schedule_excel(self, schedule: Dict[str, List[int]], out_path: Path):
        """
        Save a schedule dict {resource_name: [0/1,...]} to an Excel that matches
        the 'General' sheet format you read at startup (Resource + time-slot columns).
        """
        try:
            # Reuse the same time-slot columns as the initial schedule, if present
            if hasattr(self, "_schedule_df") and len(self._schedule_df.columns) > 0:
                cols = list(self._schedule_df.columns)
            else:
                # Fallback generic names
                max_len = max((len(v) for v in schedule.values()), default=0)
                cols = [f"t{i+1}" for i in range(max_len)]

            df = pd.DataFrame(schedule).T
            df.index.name = "Resource"
            df.columns = cols

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(out_path) as writer:
                df.to_excel(writer, sheet_name="General")

                # Optional KPI summary sheet (if we captured them)
                if self.ga_best_metrics is not None:
                    rc_avg, lead_min, rel = self.ga_best_metrics
                else:
                    rc_avg = lead_min = rel = None

                meta = pd.DataFrame({
                    "metric": [
                        "ga_best_score",
                        "utilization_avg",
                        "lead_time_min",
                        "reliability",
                        "num_generations",
                        "population_size",
                        "mutation_rate",
                    ],
                    "value": [
                        self.ga_best_score,
                        rc_avg,
                        lead_min,
                        rel,
                        self.num_generations,
                        self.population_size,
                        self.mutation_rate,
                    ],
                })
                meta.to_excel(writer, sheet_name="Summary", index=False)

            print(f"GA best schedule saved to: {out_path}")

        except Exception as e:
            print(f"Failed to export GA best schedule to Excel: {e}")

    # ------- GA internals -------

    def _generate_initial_scenario(self):
        availability = {}
        for res in self.resources:
            res_name_norm = res.name.lower().strip()
            matched_row = next(
                (row for row in self._schedule_df.index if str(row).lower().strip() == res_name_norm),
                None
            )
            if matched_row is not None:
                availability[res.name] = list(self._schedule_df.loc[matched_row].values)
            else:
                availability[res.name] = [random.randint(0, 1) for _ in range(self.num_time_slots)]

        return {
            "resources": self.resources,
            "availability": availability
        }

    def _get_trial_evaluation(self, scenario_input_parameters: Dict) -> Tuple[StateModel, float]:
        """
        Sequential evaluation used at the very end on the best schedule to get
        a final score and to optionally serialize the model once.
        """
        # Load fresh base model (do not reuse mutated self._state_model)
        state_model = deserialize_state_model(
            Path(self._sim_config["base_model_file_path"]),
            state_model_generation_settings=None,
            persistence_format="pkl",
            deserialization_required=False,
        )

        dd = self._sim_config.get("delivery_dates", None)
        if dd:
            _apply_delivery_dates(state_model, dd, verbose=self._verbose)
        _apply_availability_to_model(state_model, scenario_input_parameters["availability"], verbose=self._verbose)

        # Simulate
        port_str = str(self._sim_config["xmpp_server_rest_api_port"]).strip()
        if not port_str.startswith(":"):
            if port_str.isdigit() or ":" not in port_str:
                port_str = ":" + port_str

        starter = self._sim_config["simulation_starter"](
            project_path=Path(self._sim_config["project_path"]),
            path_to_models=self._sim_config["path_to_models"],
            xmpp_server_ip_address=self._sim_config["xmpp_server_ip_address"],
            xmpp_server_rest_api_port=port_str,
            xmpp_server_shared_secret=self._sim_config["xmpp_server_shared_secret"],
            xmpp_server_rest_api_users_endpoint=self._sim_config["xmpp_server_rest_api_users_endpoint"],
        )

        starter.simulate(
            digital_twin=state_model,
            start_time_simulation=datetime.now(),
            digital_twin_update_paths=self._scenario_settings,
            agents_file_name=self._agents_model_file,
            order_agent_amount=self.order_agent_amount,
        )

        # KPIs
        active_resource_names = [name for name, sched in scenario_input_parameters["availability"].items() if any(sched)]
        rc_util = state_model.get_resource_capacity_utilization(active_resource_names)
        lead = state_model.get_order_lead_time_mean()
        rel = state_model.get_delivery_reliability()

        if isinstance(rc_util, dict):
            values = list(rc_util.values())
            rc_avg = sum(values) / len(values) if values else 0.0
        elif isinstance(rc_util, list):
            rc_avg = sum(rc_util) / len(rc_util) if rc_util else 0.0
        else:
            rc_avg = float(rc_util or 0.0)

        if isinstance(lead, timedelta):
            lead_min = lead.total_seconds() / 60.0
        else:
            try:
                lead_min = float(lead)
            except Exception:
                lead_min = 999.0

        try:
            rel = float(rel)
        except Exception:
            rel = 0.0

        min_lead = self.required_order_lead_time_min.total_seconds() / 60.0
        max_lead = self.required_order_lead_time_max.total_seconds() / 60.0
        denom = max(max_lead - min_lead, 1e-9)
        ol_norm = (lead_min - min_lead) / denom
        if math.isnan(ol_norm):
            ol_norm = 1.0
        else:
            ol_norm = max(0.0, min(1.0, ol_norm))

        score = 1.0 * rc_avg - 0.5 * ol_norm + 2.0 * rel
        return state_model, score

    def _prepare_scenario(self, scenario_input_parameters):
        # not used in the parallel code path anymore
        pass

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        child_availability = {}
        keys = set(parent1.get("availability", {}).keys()) | set(parent2.get("availability", {}).keys())
        for res_name in keys:
            slots1 = parent1["availability"].get(res_name, [])
            slots2 = parent2["availability"].get(res_name, [])
            if not slots1:
                child_availability[res_name] = list(slots2)
            elif not slots2:
                child_availability[res_name] = list(slots1)
            else:
                child_availability[res_name] = [random.choice([a, b]) for a, b in zip(slots1, slots2)]
        return {"availability": child_availability}

    def _mutate(self, individual: Dict) -> Dict:
        if not individual.get("availability"):
            return individual
        res_name = random.choice(list(individual["availability"].keys()))
        if not individual["availability"][res_name]:
            return individual
        idx = random.randint(0, len(individual["availability"][res_name]) - 1)
        individual["availability"][res_name][idx] ^= 1
        return individual
