"""
Provides information about the availability of resources and parts to execute a process.
"""
# Imports Part 1: Standard Imports
from __future__ import annotations

import asyncio
from datetime import timedelta, datetime
from functools import reduce
from operator import concat
from typing import TYPE_CHECKING, Optional, List, Dict, Union

# Imports Part 2: PIP Imports
import numpy as np
from numpy.ma.core import round_

# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour
from ofact.twin.agent_control.behaviours.planning.tree.preference import _get_merged_time_periods_vertical
from ofact.twin.agent_control.helpers.communication_objects import DictCO
from ofact.twin.agent_control.helpers.debug_str import get_debug_str
from ofact.twin.state_model.entities import Resource
from ofact.helpers import convert_to_datetime

if TYPE_CHECKING:
    from ofact.twin.state_model.processes import Process, ValueAddedProcess
    from ofact.twin.state_model.model import StateModel


def get_earliest_possible_execution_times_dict(process_entities_groups, entity_type_entities,
                                               entities_free_calendar_extracts, process_estimated_lead_time):
    return {process: get_earliest_possible_execution(required_entities_groups, entity_type_entities,
                                                     entities_free_calendar_extracts,
                                                     process_estimated_lead_time[process])
            for process, required_entities_groups in process_entities_groups.items()}



def get_earliest_possible_execution(required_entities_groups, entity_type_entities, entities_free_calendar_extracts,
                                    estimated_lead_time) -> Optional[timedelta]:
    earliest_time_stamp_best = None
    for required_process_group in required_entities_groups:

        earliest_time_stamp_process_group = \
            get_earliest_execution_for_resource_group(required_process_group, entity_type_entities,
                                                      entities_free_calendar_extracts, estimated_lead_time)

        if earliest_time_stamp_best is None:
            earliest_time_stamp_best = earliest_time_stamp_process_group
        elif earliest_time_stamp_best > earliest_time_stamp_process_group:
            earliest_time_stamp_best = earliest_time_stamp_process_group

    return earliest_time_stamp_best


def get_earliest_execution_for_resource_group(required_process_group, entity_type_entities,
                                              entities_free_calendar_extracts, estimated_lead_time):
    process_group_free_calendar_extracts = \
        get_process_group_free_calendar_extracts(required_process_group, entity_type_entities,
                                                 entities_free_calendar_extracts)

    overlaps = _get_merged_time_periods_vertical(process_group_free_calendar_extracts)

    # determine overlaps that have the estimated lead time
    mask = np.where(overlaps[:, 1] - overlaps[:, 0] >= np.timedelta64(int(estimated_lead_time), "s"))
    possible_merged_time_periods = overlaps[mask]

    if not possible_merged_time_periods.size:
        return None

    earliest_time_stamp_process_group = convert_to_datetime(possible_merged_time_periods[0, 0])

    return earliest_time_stamp_process_group


def get_process_group_free_calendar_extracts(required_process_group, entity_type_entities,
                                             entities_free_calendar_extracts):
    process_group_free_calendar_extracts = []
    for entity_type in required_process_group:
        entities = entity_type_entities[entity_type]

        start_time_stamp = None
        earliest_free_calendar_extract = None
        for entity in entities:  # Pruning needed (as standard procedure ...)

            if entity not in entities_free_calendar_extracts:
                continue

            free_calendar_extract = entities_free_calendar_extracts[entity]

            # consider only one resource - problem if the time available to short for execution the process needed
            if free_calendar_extract.any():
                if start_time_stamp is None:
                    earliest_free_calendar_extract = free_calendar_extract
                    start_time_stamp = free_calendar_extract[0, 0]

                elif free_calendar_extract[0, 0] < start_time_stamp:
                    earliest_free_calendar_extract = free_calendar_extract
                    start_time_stamp = free_calendar_extract[0, 0]

        if earliest_free_calendar_extract is not None:
            process_group_free_calendar_extracts.append(earliest_free_calendar_extract)

        return process_group_free_calendar_extracts


class AvailableTimeSlotService(DigitalTwinCyclicBehaviour):
    """
    The available time slot service is requested if an order agent wants to know,
    when his process can be processed next, based on the entities needed.
    Therefore, it is looked up, when the required entities are available for the duration of the process lead time
    at the same time.
    """

    availability_template = {"metadata": {"performative": "request",
                                          "ontology": "AVAILABILITY",
                                          "language": "OWL-S"}}
    templates = [availability_template]

    def __init__(self):
        """

        :attribute process_resources_to_consider: assuming that the consideration of bottleneck resource is sufficient
        """

        super(AvailableTimeSlotService, self).__init__()

        self.metadata_conditions = {'performative': 'request', 'ontology': 'AVAILABILITY'}
        self.process_resource_entity_types_to_consider: Dict[Union[Process, ValueAddedProcess], List[Resource]] = {}
        self.request_before = {"round": 0,
                               "requester": []}

        self.current_requests = {}
        self.last_print = datetime.now()

        self.start_time_stamp = None
        self.processes = {}
        self.tasks_senders = {}

    def set_initial(self, digital_twin: StateModel):
        pass  # can be overwritten

    async def run(self):
        await super().run()

        msg_received = await self.agent.receive_msg(self, timeout=10, metadata_conditions=self.metadata_conditions)

        if msg_received is None:
            return

        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" Requesting availability information!")

        if not self.tasks_senders:
            asyncio.create_task(self._send_information())

        msg_content, msg_sender, msg_ontology, msg_performative = msg_received
        processes, start_time_stamp, round = msg_content
        self.tasks_senders.setdefault(round,
                                      []).append(msg_sender)

        self.processes.setdefault(round,
                                  []).extend(processes)

        if self.start_time_stamp is None:
            self.start_time_stamp = start_time_stamp
        elif start_time_stamp < self.start_time_stamp:
            self.start_time_stamp = start_time_stamp

    async def _send_information(self):

        await asyncio.sleep(.1)

        start_time_stamp, self.start_time_stamp = self.start_time_stamp, None
        processes_round, self.processes = self.processes, {}
        tasks_senders, self.tasks_senders = self.tasks_senders, {}

        awaits = []
        processes_to_process = []
        processes_to_await = []

        for round, processes in processes_round.items():
            processes = list(set(processes))
            if round not in self.current_requests:
                self.current_requests[round] = {}

            for process in processes:
                if process in self.current_requests[round]:
                    awaits.append(self.current_requests[round][process][0])
                    processes_to_await.append(process)
                else:
                    processes_to_process.append(process)
                    self.current_requests[round][process] = (asyncio.get_running_loop().create_future(),
                                                             0)

            earliest_possible_execution_times = {}
            if processes_to_process:
                earliest_possible_execution_times = \
                    self.get_earliest_possible_execution_times(processes_to_process, self.agent.digital_twin,
                                                               np.datetime64(start_time_stamp),
                                                               np.timedelta64(3, "D"))
                for process, time_ in earliest_possible_execution_times.items():
                    self.current_requests[round][process] = (self.current_requests[round][process][0], time_)
                    self.current_requests[round][process][0].set_result(True)

            if processes_to_await:
                for process in processes_to_await:
                    await self.current_requests[round][process][0]

                earliest_possible_execution_times |= {process: self.current_requests[round][process][1]
                                                      for process in processes_to_await}

            earliest_possible_execution_time_minimum = min(list(earliest_possible_execution_times.values()))

            for msg_sender in tasks_senders[round]:
                if msg_sender in self.request_before:
                    if self.request_before[msg_sender] >= earliest_possible_execution_time_minimum:
                        earliest_possible_execution_times = \
                            self.get_earliest_possible_execution_times(
                                processes, self.agent.digital_twin,
                                np.datetime64(start_time_stamp + timedelta(seconds=180)),
                                np.timedelta64(3, "D"))
                        earliest_possible_execution_time_minimum = min(list(earliest_possible_execution_times.values()))

                self.request_before[msg_sender] = earliest_possible_execution_time_minimum

                if self.last_print + timedelta(seconds=1) < datetime.now():
                    print(get_debug_str(self.agent.name, self.__class__.__name__) +
                          f" Send availability information {min(list(earliest_possible_execution_times.values()))}")
                self.last_print = datetime.now()
                # print("Information Service:", {vap.name: time for vap, time in earliest_possible_execution_times.items()})
                msg_content = DictCO(content_dict=earliest_possible_execution_times)

                await self.agent.send_msg(behaviour=self, receiver_list=[msg_sender], msg_body=msg_content,
                                          message_metadata={"performative": "inform-result",
                                                            "ontology": "AVAILABILITY",
                                                            "language": "OWL-S"})

    def get_earliest_possible_execution_times(self, processes: List[Process], digital_twin: StateModel,
                                              start_time_stamp: np.datetime64, duration_considered):
        """
        Determine the time, a process could be executed based on the schedules of the resources that participate in the
        process.
        ToDo: consider the transport of the resources to the resources and so on ...
         currently it is only a heuristic approach
        :param processes: a list of processes that could be executed
        :param digital_twin: the digital twin model provides the resources and therefore the schedules
        :param start_time_stamp: start_time_stamp
        :return: a dictionary of processes and their earliest possible execution
        """

        relevant_entity_types, process_entities_groups, process_estimated_lead_time = (
            self.get_process_parameters(processes))

        # maybe approach of a resource needed
        process_lead_time_min = np.timedelta64(int(min(list(process_estimated_lead_time.values()))), "s")

        entity_type_entities = digital_twin.get_entities_by_entity_types(list(relevant_entity_types))
        entities = list(set(reduce(concat, list(entity_type_entities.values()))))

        entities_free_calendar_extracts = {}
        end_time = start_time_stamp + duration_considered

        for entity in entities:
            if not isinstance(entity, Resource):
                continue
            # ToDo: consider only the parts ...
            try:
                free_periods = entity.get_free_periods_calendar_extract(start_time=start_time_stamp,
                                                                    end_time=end_time,
                                                                    time_slot_duration=process_lead_time_min)
            except:
                continue
            entities_free_calendar_extracts[entity] = free_periods

        earliest_possible_execution_times = (
            get_earliest_possible_execution_times_dict(process_entities_groups, entity_type_entities,
                                                       entities_free_calendar_extracts, process_estimated_lead_time))

        return earliest_possible_execution_times

    def get_process_parameters(self, processes):
        relevant_entity_types = set()
        process_entities_groups = {}
        process_estimated_lead_time = {}
        for process in processes:
            if process in self.process_resource_entity_types_to_consider:
                # restriction of the consideration frame for better performance based on prior knowledge
                entities_required_groups = self.process_resource_entity_types_to_consider[process]
            else:
                # transport resources should also be considered
                entities_required_groups = process.get_entities_needed()
            process_entities_groups[process] = entities_required_groups
            relevant_entity_types |= set(reduce(concat, entities_required_groups))

            process_estimated_lead_time[process] = self._get_process_lead_time(process)

        return relevant_entity_types, process_entities_groups, process_estimated_lead_time

    def _get_process_lead_time(self, process):
        estimated_process_lead_time = process.get_estimated_process_lead_time()
        return estimated_process_lead_time
