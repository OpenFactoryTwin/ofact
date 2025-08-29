"""
This file encapsulates the participant of the negotiation that respond to the planning of the initiator.
http://www.fipa.org/specs/fipa00037/SC00037J.html#_Toc26729693
"""
# Imports Part 1: Standard Imports
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import timedelta, datetime
from typing import TYPE_CHECKING, Dict

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour
from ofact.twin.agent_control.behaviours.env_release.process_executions_queue import (
    ProcessExecutionsOnHoldHandling)
from ofact.twin.agent_control.behaviours.negotiation.objects import (
    ResourceCallForProposal, Proposal, CallForProposal, PartCallForProposal, ProcessCallForProposal,
    ProcessGroupCallForProposal, ProcessExecutionsPathProposal)
from ofact.twin.agent_control.behaviours.planning.tree.process_executions_components import (
    ProcessExecutionsComponent)
from ofact.twin.agent_control.helpers.communication_objects import CoordinationCO, ListCO, ObjectCO
from ofact.twin.agent_control.helpers.debug_str import get_debug_str

from ofact.twin.utils import setup_dual_logger
logging=setup_dual_logger()
if TYPE_CHECKING:
    pass

# Module-Specific Constants
#logger = logging.getLogger("CNETParticipant")


def _get_proposals_to_schedule(requests):
    """Determines the proposals of the lowest level"""
    proposals_to_schedule = [proposal_to_schedule
                             for cfp, proposals in requests.items()
                             for proposal_to_schedule in _get_elements_to_schedule_sub(cfp, proposals)
                             if proposal_to_schedule.status == Proposal.Status.OPEN]

    proposals_to_reject = [proposal_to_schedule
                           for cfp, proposals in requests.items()
                           for proposal_to_schedule in _get_elements_to_schedule_sub(cfp, proposals)
                           if proposal_to_schedule.status == Proposal.Status.REJECTED]

    return proposals_to_schedule, proposals_to_reject


def _get_elements_to_schedule_sub(cfp, proposals):
    """Determines the proposals of the lowest level"""
    # PartProposals that did not have sub_proposals

    return [proposal for proposal in proposals
            if isinstance(cfp, ResourceCallForProposal) or
            not proposal.get_sub_proposals()]


def _get_lowest_level_elements(elements_to_schedule):
    """
    Sort all process_executions_component towards frontier of the tree and aggregation of the frontier leaves
    :param elements_to_schedule: all process_executions_components (only the frontier is scheduled)
    :return: a separation in lowest_level_element and highest_level_elements
    """

    lowest_level_elements = []
    highest_level_elements = []
    for proposal_to_schedule in elements_to_schedule:

        if isinstance(proposal_to_schedule, ProcessExecutionsPathProposal):

            lower_level_elements, higher_level_elements = _determine_component_level(proposal_to_schedule)

            lowest_level_elements.extend(lower_level_elements)
            highest_level_elements.extend(higher_level_elements)

        else:
            highest_level_elements.append(proposal_to_schedule)

    return lowest_level_elements, highest_level_elements


def _determine_component_level(component_to_schedule):
    """
    Determines the component level for the component_to_schedule and for his lower leaves. (one level deeper)
    :param component_to_schedule: a process_executions_component
    :return: two lists (sorted in lower and higher level)
    """

    lower_level_elements = []
    higher_level_elements = []
    component_not_set = True
    if not component_to_schedule.process_executions_components:
        lower_level_elements.append(component_to_schedule)
        component_not_set = False

    for process_executions_component in component_to_schedule.get_process_executions_components_lst():
        if process_executions_component.cfp_path != component_to_schedule.cfp_path:
            higher_level_elements.append(component_to_schedule)

        for lower_process_executions_component in process_executions_component.get_process_executions_components_lst():
            if lower_process_executions_component.process_executions_components:
                continue
            if component_not_set:
                lower_level_elements.append(component_to_schedule)
                component_not_set = False
            lower_level_elements.append(lower_process_executions_component)

    return lower_level_elements, higher_level_elements


def _get_proposals_by_process_executions_components(proposals_to_schedule, alternatives):
    proposals = list(set(alternatives).intersection(proposals_to_schedule))
    additional_components = list(set(alternatives).difference(proposals_to_schedule))
    return proposals, additional_components


class ParticipantBehaviour(DigitalTwinCyclicBehaviour):
    """
    The participant behaviour iterates through the run method ...
    The participant is round progress dependent. Therefore, to the beginning it syncs with the other agents
    (round start).
    With the start of a round, it hears and collect call for proposals (cfp)/ requests (planning).
    If a new cfp is available, the planning process is triggered.
    If all call for proposals are available a second sync is done. <br/>
    The agents that have open proposals, go into the end negotiation phase.
    In the end negotiation, the proposals that should be scheduled (resource and part) are forwarded to the coordinator
    agent. Getting back the scheduled proposals/ components, the results are passed to the requesters,
    until the first requester is reached.
    The first requester has the option to reject or accept the proposal received or to receive a rejection in response
    which does not entail any possibility.
    The acceptance or rejection is afterward passed to the agents and the process_executions are passed
    to the environment.
    """

    request_cfp_template = {"metadata": {"performative": "cfp",
                                         "ontology": "CFP",
                                         "language": "OWL-S"}}
    rejected_proposal_template = {"metadata": {"performative": "reject-proposal",
                                               "ontology": "PLANNING_PROPOSAL",
                                               "language": "OWL-S"}}
    accepted_proposal_template = {"metadata": {"performative": "accept-proposal",
                                               "ontology": "PLANNING_PROPOSAL",
                                               "language": "OWL-S"}}
    coordination_inform_result_template = {"metadata": {"performative": "inform-result",
                                                        "ontology": "COORDINATION",
                                                        "language": "OWL-S"}}
    templates = [request_cfp_template, rejected_proposal_template, accepted_proposal_template,
                 coordination_inform_result_template]

    def __init__(self, collection_phase_period: float, collection_volume_limit: int, connection_behaviour):
        super(ParticipantBehaviour, self).__init__()

        self.connection_behaviour = connection_behaviour

        self._round = None

        self.metadata_conditions_lst = [{"performative": "cfp", "ontology": "CFP"},
                                        {"performative": "reject-proposal", "ontology": "PLANNING_PROPOSAL"},
                                        {"performative": "accept-proposal", "ontology": "PLANNING_PROPOSAL"},
                                        {"performative": "inform-result", "ontology": "COORDINATION"}]

        self.requests_to_process: dict[int, dict] = {}

        # first round
        self.requests_met = {}
        # shelved means that these planning have no reference_cfp -
        # scheduling needed before for the requests_met

        self._coordination_result_object = None
        self._coordination_finished = asyncio.Event()

        self.requests_shelved = {}
        self.requests_ready_for_proposal = {}
        self.requests_ready_for_refuse = {}

        self._ready_for_proposal_creation = asyncio.Event()
        self.proposals_evaluated: dict[CallForProposal, list[Proposal]] = {}  # evaluated after scheduling

        self.open_proposals = {}

        self.proposals_to_accept = {}

        self.accepted_proposals = []
        self.rejected_proposals = []

        self.negotiation_completed = {}  # negotiation_object_identification -
        # future that is set to finished if all responses available
        self.finishing_opened = asyncio.Event()
        self.round_finished = None

        self.second = False  # debugging

        self.request_method_mapper = None
        self.process_executions_on_hold_handling = None

    async def on_start(self):
        await super().on_start()

        self.process_executions_on_hold_handling = ProcessExecutionsOnHoldHandling(self.agent)

        self.request_method_mapper = {}

        if hasattr(self.agent, "ProcessRequest"):
            self.request_method_mapper[ProcessCallForProposal] = self.agent.ProcessRequest.process_request
        if hasattr(self.agent, "ResourceRequest"):
            self.request_method_mapper[ResourceCallForProposal] = self.agent.ResourceRequest.process_request
        if hasattr(self.agent, "PartRequest"):
            self.request_method_mapper[PartCallForProposal] = self.agent.PartRequest.process_request
        if hasattr(self.agent, "ProcessGroupRequest"):
            self.request_method_mapper[ProcessGroupCallForProposal] = self.agent.ProcessGroupRequest.process_request

    @classmethod
    def get_templates(cls):
        return deepcopy(cls.templates)

    async def run(self):
        await super(ParticipantBehaviour, self).run()

        # logger.debug(get_debug_str(self.agent.name, self.__class__.__name__) + f" Begin with negotiation round")
        msg_receiving_task = self.get_hearing_task()
        await self._sync()  # explicit sync
        # logger.debug(get_debug_str(self.agent.name, self.__class__.__name__) + f" Requester ready")

        # service_provider_round
        # determine, if all call for proposals can be met (respectively wait until all proposals come in
        # for the determination)
        task_received = await self._process_msg(msg_receiving_task)
        # logger.debug(get_debug_str(self.agent.name, self.__class__.__name__) +
        #              f"Shelved {self.requests_shelved}, Met {self.requests_met}")

        # implicit sync
        if task_received:
            self.agent.EnvInterface.pre_reaction = False
            self.agent.agents.in_end_negotiation(self._round, self.agent.name)
            self.round_finished = asyncio.Event()
            task = asyncio.create_task(self._end_negotiation())  # task to run it in parallel

            await self._process_msg(phase_1=False)
            await task
            await self.process_executions_on_hold_handling.request_process_executions_available(round_ended=True)

        else:  # if "OrderDigitalTwinAgent" not in self.agent.__class__.__name__
            if self.agent.agents.get_current_round()[1] == self.agent.agents.max_round_cycles - 1:
                if self.agent.reaction_expected and not self.agent.process_executions_on_hold:
                    self.agent.reaction_expected = False
                    await self.agent.change_handler.go_on(agent_name=self.agent.name)

                else:
                    # no reaction expected answer needed if requested ...
                    self.agent.EnvInterface.pre_reaction = True

            self.agent.agents.not_in_end_negotiation(self._round, self.agent.name)

    async def _sync(self):
        """Sync rounds of negotiation - therefore each agent starts at the same time with a new round"""
        if "OrderDigitalTwinAgent" not in self.agent.__class__.__name__:
            # the order agents make their sync in the order_management behaviour!
            self._round, skip_complete_round = await self.agent.agents.ready_for_next_round(self.agent.name)

        else:
            self._round = await self.agent.agents.wait_on_other_requesters()

        # print(self.agent.name, self._round)

    def get_hearing_task(self):
        hearing_task = asyncio.create_task(self.agent.receive_msg(self, timeout=10,
                                                                  metadata_conditions_lst=self.metadata_conditions_lst))

        return hearing_task

    async def _process_msg(self, msg_receiving_task=None, phase_1=True):
        """Process the message receiving for the behaviour - it can handle different incoming messages"""
        task_received = False
        if phase_1:
            phase_end_task = asyncio.create_task(self.agent.agents.get_planning_proposal_ended_status(self._round))
        else:
            phase_end_task = asyncio.create_task(self.get_round_finished_status())

        if msg_receiving_task is None:
            msg_receiving_task = self.get_hearing_task()

        while True:
            tasks = [msg_receiving_task, phase_end_task]
            done_tasks, pending_tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for done_task in done_tasks:
                task_result = done_task.result()
                if isinstance(task_result, tuple):
                    msg_content, msg_sender, msg_ontology, msg_performative = task_result
                    background_task = asyncio.create_task(self._handle_received_msg(msg_content, msg_sender,
                                                                                      msg_ontology, msg_performative))
                    task_received = True

                elif task_result is not True:
                    continue

                else:
                    # it is important to kill the existing tasks because alternatively lost the possible result of them
                    for pending_task in pending_tasks:
                        pending_task.cancel()

                    # print(self.agent.name, datetime.now(), phase_1)
                    return task_received

            msg_receiving_task = self.get_hearing_task()

    async def _handle_received_msg(self, msg_content, msg_sender, msg_ontology, msg_performative):
        """Handle the received messages for the behaviour,
        respectively forward the messages to the appropriate methods"""

        negotiation_objects = msg_content
        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" negotiation_objects: {negotiation_objects}")
        # print(get_debug_str(self.agent.name, self.__class__.__name__) + "Got message")
        if msg_ontology == "CFP":
            # append the new request to the collection of the negotiation_behaviour
            await self.collect_requests(negotiation_objects=negotiation_objects)

        elif msg_ontology == "PLANNING_PROPOSAL":
            # logger.debug(f"{datetime.now().time()} | {msg_received.metadata['performative']} received")

            if msg_performative == "reject-proposal":
                await self.react_on_rejected_proposal(negotiation_objects, msg_sender)

            elif msg_performative == "accept-proposal":
                await self.react_on_accepted_proposal(negotiation_objects, msg_sender)

        elif msg_ontology == "COORDINATION" and msg_performative == "inform-result":
            # print(self.agent.name, msg_content)
            await self._react_on_coordination_response(negotiation_objects)

    async def collect_requests(self, negotiation_objects):
        """
        Manage the request collection
        :param negotiation_objects: a list of negotiation objects - request
        that are necessary to perform the process
        """
        for negotiation_object in negotiation_objects:
            # record the request in a standardized format

            same_objects = bool([request for neg_obj_id, request in self.requests_to_process.items()
                                 if neg_obj_id == negotiation_object.identification])
            if same_objects:
                continue

            request_information = {"negotiation_object": negotiation_object}
            self.requests_to_process[negotiation_object.identification] = request_information
            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       f" ID: '{negotiation_object.identification}' added from {negotiation_object.sender_name}")

            task_planning_request = asyncio.create_task(self._process_planning_request(request_information))

    async def _process_planning_request(self, request_dict):
        """Process the request for the current phase (after the end of the collection_phase)"""

        # logger.debug(get_debug_str(self.agent.name, self.__class__.__name__) + f" request_processing: {request_dict}")

        # ensure preconditions
        successful, process_executions_components, proposals_to_reject = \
            await self.create_process_executions_paths_variants(request_dict["negotiation_object"])

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Successful: ", successful)

        if proposals_to_reject:
            # print("Proposals to reject: ", [p.identification for p in proposals_to_reject])
            proposal_to_reject_new = []
            proposals_batch = proposals_to_reject.copy()
            proposals_batch_new = []
            proposals_to_reject_lowest_level = []
            while proposals_batch:
                for proposal in proposals_batch:
                    if proposal.overarching_proposal_rejected():
                        proposal_to_reject_new.append(proposal)
                        proposals_batch_new.extend(proposal.get_sub_proposals())
                        if not proposal.get_sub_proposals():
                            proposals_to_reject_lowest_level.append(proposal)
                proposals_batch = proposals_batch_new.copy()
                proposals_batch_new = []
            await self._send_refinement_updates(proposals_to_reject_lowest_level, [])  # ToDo: not really decentral
            await self._handle_rejected_proposals(proposals_to_reject)

        request_dict = await self._process_request_results(successful, request_dict, process_executions_components)

        if request_dict["negotiation_object"].reference_cfp is None:
            first_order_cfp = True
        else:
            first_order_cfp = False  # determines if the agent is responsible for first_order_cfps

        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" Delete requests_to_process: {request_dict['negotiation_object'].identification}")
        del self.requests_to_process[request_dict["negotiation_object"].identification]

        if first_order_cfp:
            # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" set_planning_proposal_ended")
            await self.agent.agents.set_planning_proposal_ended(self.agent.name, self._round)

            if not self.requests_to_process:
                self.agent.agents.set_planning_proposal_ended_eventually(self._round)

    async def _process_request_results(self, successful, request_dict, process_executions_components_proposals):
        """Process the results coming from planning  (tree)"""

        if not successful:
            # no proposal can be made because the administrated resources cannot perform the process or
            # did not want it
            await self._send_refuse(sender=request_dict["negotiation_object"].sender_name,
                                    negotiation_object=request_dict["negotiation_object"])

            return request_dict

        # based on the ensured preconditions a proposal is created (if ensurance was successful)
        # print(get_debug_str(self.agent.name, self.__class__.__name__) + " Proposals")

        if request_dict["negotiation_object"].reference_cfp is None:
            # print(self.agent.name, "s stored", request_dict["negotiation_object"].identification)
            self.requests_shelved[request_dict["negotiation_object"]] = process_executions_components_proposals

        else:
            # the proposal is sent to the requester
            await self._send_proposals(sender=request_dict["negotiation_object"].sender_name,
                                       proposals=process_executions_components_proposals)

            # print(self.agent.name, "n stored", request_dict["negotiation_object"].identification)
            # print("Request identification: ", request_dict["negotiation_object"].identification)
            # print("Length proposals:", request_dict["negotiation_object"].identification,
            #       len(process_executions_components_proposals))
            self.requests_met[request_dict["negotiation_object"]] = process_executions_components_proposals
            self.add_open_proposals(process_executions_components_proposals)

        return request_dict

    async def create_process_executions_paths_variants(self, negotiation_object_request: CallForProposal):
        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" Client {negotiation_object_request.client_object.identification} "
        #       f"has started the request - {negotiation_object_request.request_object}")

        if negotiation_object_request.__class__ not in self.request_method_mapper:
            raise NotImplementedError("Request type unknown")

        request_func = self.request_method_mapper[negotiation_object_request.__class__]

        successful, process_executions_variants, proposals_to_reject = \
            await request_func(negotiation_object_request)

        return successful, process_executions_variants, proposals_to_reject

    def _create_proposals_coordination(self, alternatives_coordination_new):
        """Create new proposal based on process_executions_components coming from the coordinator_agent"""

        if alternatives_coordination_new:
            raise NotImplementedError

        proposals_coordination = []
        return proposals_coordination

    async def _send_proposals(self, sender, proposals):
        # store the negotiation_object in the agent_model
        list_communication_object = ListCO(proposals)
        msg_content = list_communication_object

        # object not json serializable
        await self.agent.send_msg(behaviour=self, receiver_list=[sender], msg_body=msg_content,
                                  message_metadata={"performative": "propose",
                                                    "ontology": "PLANNING_PROPOSAL",
                                                    "language": "OWL-S"})

    async def _send_refuse(self, sender, negotiation_object):
        """Refuse a call for proposal because no proposal can be built or is not wished to build"""
        communication_object = ObjectCO(negotiation_object)
        msg_content = communication_object

        # object not json serializable
        await self.agent.send_msg(behaviour=self, receiver_list=[sender], msg_body=msg_content,
                                  message_metadata={"performative": "refuse",
                                                    "ontology": "PLANNING_PROPOSAL",
                                                    "language": "OWL-S"})

    async def _handle_rejected_proposals(self, proposals_to_reject):
        """Handle rejected proposals that come from the planning structure"""
        rejected_proposals_provider = {}
        proposals_to_reject_self = []

        # print("Reject ...", len(set(proposals_to_reject)),
        #       [proposal.call_for_proposal.reference_cfp.identification
        #        if proposal.call_for_proposal.reference_cfp
        #        else None
        #        for proposal in proposals_to_reject])

        self.remove_proposals_request_met(proposals_to_reject)
        for proposal in proposals_to_reject:
            if not proposal.overarching_proposal_rejected():
                continue

            if proposal.provider == self.agent.name:
                proposals_to_reject_self.append(proposal)
            else:
                rejected_proposals_provider.setdefault(proposal.provider,
                                                       []).append(proposal)
            proposal.reject()

        if proposals_to_reject_self:
            await self.react_on_rejected_proposal(proposals_to_reject_self, self.agent.name)

        if rejected_proposals_provider:
            await self.connection_behaviour.reject_proposals_provider(rejected_proposals_provider)

    async def react_on_rejected_proposal(self, rejected_proposals, sender):
        """React on a rejected proposal by contacting the participant agents"""
        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" React on reject: {[rejected_proposal.identification for rejected_proposal in rejected_proposals]}")

        rejected_proposals = list(set(rejected_proposals))
        await self.reject_sub_proposals(rejected_proposals)

        new_rejected_proposals = [(sender, rejected_proposal)
                                  for rejected_proposal in rejected_proposals
                                  if rejected_proposal.type == ProcessExecutionsComponent.Type.STANDARD]
        self.rejected_proposals += new_rejected_proposals

        if new_rejected_proposals:
            if self.round_finished is not None:
                if not self.round_finished.is_set():
                    await self.process_executions_on_hold_handling.request_process_executions_available()

        # proposals_to_schedule = self.accepted_proposals.copy()
        self.remove_open_proposals(rejected_proposals, end=True)

        # if not self.open_proposals and proposals_to_schedule:
        #     self._handle_reservations(proposals_to_schedule)
        for proposal in rejected_proposals:
            if proposal.identification in self.proposals_to_accept:
                print(get_debug_str(self.agent.name, self.__class__.__name__) + " Rejected")
                del self.proposals_to_accept[proposal.identification]

        if not self.proposals_to_accept:
            if self.round_finished is not None:
                if not self.round_finished.is_set() and self.finishing_opened.is_set():
                    self.round_finished.set()
                    # print(self.agent.name, "Set result rej")

        # print(self.agent.name, "Wait on acceptance:", [(proposal.cfp_path, proposal.status)
        #                                                for id, proposal in self.proposals_to_accept.items()])
        return rejected_proposals

    async def reject_sub_proposals(self, rejected_proposals):
        """Reject the sub proposals"""

        supporter_proposals = {}
        for rejected_proposal in rejected_proposals:
            for proposal in rejected_proposal.get_sub_proposals():
                if not proposal.overarching_proposal_rejected():
                    continue

                proposal.reject()

                supporter_proposals.setdefault(proposal.provider,
                                               []).append(proposal)

        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" Rejected sub proposals "
        #       f"{[proposal.identification for _, proposals in supporter_proposals.items() for proposal in proposals]}")

        await self.connection_behaviour.reject_proposals_provider(supporter_proposals)

    async def react_on_accepted_proposal(self, accepted_proposals, sender):
        """React on an accepted proposal"""
        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" React on acceptance {[proposal.identification for proposal in accepted_proposals]}")
        for proposal in accepted_proposals:
            logging.debug(
                f'Order ID:{proposal.call_for_proposal.order.identification}, {proposal.identification} start react_on_accepted_proposal',
                extra={"obj_id": self.agent.name})
        # proposals in the last instance (first call for proposal)
        proposals_process_executions_creation = [accepted_proposal
                                                 for accepted_proposal in accepted_proposals
                                                 if accepted_proposal.call_for_proposal.reference_cfp is None]

        if proposals_process_executions_creation:
            # process_executions are based on the process_executions_components
            await self._send_process_executions(proposals_process_executions_creation)
        logging.debug(
            f'Order ID:{proposal.call_for_proposal.order.identification}, {proposal.identification} await 1 react_on_accepted_proposal',
            extra={"obj_id": self.agent.name})
        # Choose the best fitting time_slot
        await self.accept_sub_proposals(accepted_proposals)
        logging.debug(
            f'Order ID:{proposal.call_for_proposal.order.identification}, {proposal.identification} await 2 react_on_accepted_proposal',
            extra={"obj_id": self.agent.name})

        # record the time_slot into the resource calendars
        new_accepted_proposals = [(sender, accepted_proposal)
                                  for accepted_proposal in accepted_proposals
                                  if accepted_proposal.type == ProcessExecutionsComponent.Type.STANDARD]
        self.accepted_proposals += new_accepted_proposals
        if new_accepted_proposals:
            if self.round_finished is not None:
                if not self.round_finished.is_set():
                    # print(self.agent.name, "A", [a.identification for a in accepted_proposals], self.round_finished)
                    logging.debug(
                        f'Order ID:{proposal.call_for_proposal.order.identification}, {proposal.identification}  in new accepted_proposal react_on_accepted_proposal',
                        extra={"obj_id": self.agent.name})

                    await self.process_executions_on_hold_handling.request_process_executions_available(order_id=proposal.call_for_proposal.order.identification)
                    # print(self.agent.name, "B", [a.identification for a in accepted_proposals], self.round_finished)
                    logging.debug(
                        f'Order ID:{proposal.call_for_proposal.order.identification}, {proposal.identification}  out new accepted_proposal react_on_accepted_proposal',
                        extra={"obj_id": self.agent.name})
        # proposals_to_schedule = self.accepted_proposals.copy()
        # should be handled based on the process_executions_occur
        self.remove_open_proposals(accepted_proposals, end=True)
        # if not self.open_proposals and proposals_to_schedule:
        #     self._handle_reservations(proposals_to_schedule)
        for proposal in accepted_proposals:
            # print(proposal.identification, "accepted")
            if proposal.identification in self.proposals_to_accept:
                del self.proposals_to_accept[proposal.identification]
        logging.debug(
            f'Order ID:{proposal.call_for_proposal.order.identification}, {proposal.identification} Round finished could set',
            extra={"obj_id": self.agent.name})
        if not self.proposals_to_accept:
            if self.round_finished is not None:
                if not self.round_finished.is_set() and self.finishing_opened.is_set():
                    # print(self.agent.name, "Set result acc")
                    self.round_finished.set()
                    logging.debug(f'Order ID:{proposal.call_for_proposal.order.identification}, {proposal.identification} Round finished is set', extra={"obj_id": self.agent.name})

        # print(self.agent.name, "Wait on acceptance:", [(proposal.cfp_path, proposal.status)
        #                                                for id, proposal in self.proposals_to_accept.items()])

        return accepted_proposals

    async def accept_sub_proposals(self, accepted_proposals):
        """Accept sub_proposals"""

        supporter_proposals = {}
        for accepted_proposal in accepted_proposals:
            for proposal in accepted_proposal.get_sub_proposals():  # ToDo: only the open ones?
                if proposal.status == Proposal.Status.OPEN:
                    if proposal.acceptable():
                        proposal.accept()

                if proposal.status == Proposal.Status.ACCEPTED and proposal.type.name != "CONNECTOR":
                    supporter_proposals.setdefault(proposal.provider, []).append(proposal)

        await self.connection_behaviour.accept_proposals_provider(supporter_proposals)

    def _handle_reservations(self, proposals_to_schedule):
        logging.debug(f'handle_reservations 1', extra={"obj_id": self.agent.name})
        resources_time_slots = self._get_resources_time_slots(proposals_to_schedule)
        logging.debug(f'handle_reservations 2', extra={"obj_id": self.agent.name})
        parts_time_slots = [(accepted_proposal.goal_item, provider,
                             accepted_proposal.call_for_proposal.issue_id,
                             accepted_proposal.get_final_time_slot())
                            for provider, accepted_proposal in proposals_to_schedule
                            if isinstance(accepted_proposal.call_for_proposal, PartCallForProposal)]
        logging.debug(f'handle_reservations 3', extra={"obj_id": self.agent.name})
        self.agent.reserve_time_slots(resources_time_slots)
        logging.debug(f'handle_reservations 4', extra={"obj_id": self.agent.name})
        self.agent.reserve_parts(parts_time_slots)

    def _get_resources_time_slots(self, new_accepted_proposals):
        """Create the input to schedule process in the process_executions_plans"""
        # directly creating a dataframe
        resources_time_slots = []
        for provider, accepted_proposal in new_accepted_proposals:
            if not isinstance(accepted_proposal.call_for_proposal, ResourceCallForProposal):
                continue
            # print(accepted_proposal.goal.identification, accepted_proposal.process_execution_id)
            resource = accepted_proposal.goal_item
            resources_time_slots.append((resource, provider,
                                         accepted_proposal.process_execution_id,
                                         accepted_proposal.call_for_proposal.order.identification,
                                         accepted_proposal.call_for_proposal.issue_id,
                                         accepted_proposal.get_final_time_slot()))

            # used for the transport access (find later a better solution)
            sub_components = accepted_proposal.get_process_executions_components_lst()
            if not sub_components:
                continue

            for sub_component in sub_components:
                for process_executions_component in sub_component.get_process_executions_components_lst():
                    if process_executions_component.status != Proposal.Status.ACCEPTED:
                        continue
                    resource = process_executions_component.goal_item
                    resources_time_slots.append((resource, provider,
                                                 process_executions_component.process_execution_id,
                                                 accepted_proposal.call_for_proposal.order.identification,
                                                 accepted_proposal.call_for_proposal.issue_id,
                                                 process_executions_component.get_final_time_slot()))

        return resources_time_slots

    async def _end_negotiation(self):
        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Start end_negotiation")

        proposals_to_refuse, proposals_to_update = await self._schedule(self.requests_met)

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Scheduling finished")
        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" {proposals_to_refuse}, {proposals_to_update}, {proposals_unchanged}")
        # print("Partial schedules: ", self.agent.name, [proposal.cfp_path for proposal in proposals_to_update])
        await self._send_refinement_updates(proposals_to_refuse, proposals_to_update)

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Interim negotiation"),#  self.open_proposals)
        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Interim negotiation", self.requests_shelved)

        self._ready_for_proposal_creation.clear()
        if self.requests_met or self.requests_shelved:
            # print(self.requests_met, self.requests_shelved)
            await self.ready_for_proposal_creation()
            # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Ready for proposal creation")

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Interim I negotiation")
        if self.requests_ready_for_proposal or self.requests_ready_for_refuse:
            await self._process_shelved_proposals()

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Interim II negotiation",
        #       len(self.proposals_to_accept))

        if not self.proposals_to_accept:
            self.proposals_evaluated = {}
            # print(self.agent.name, "Set result nor")
            if self.round_finished is not None:
                self.round_finished.set()
                # logging.debug(
                #     f'Order ID:{proposals_to_update[0].call_for_proposal.order.identification}:
                #     Round finished is set for return',
                #     extra={"obj_id": self.agent.name})
            return

        self.finishing_opened.set()
        # print(self.agent.name, "I")  # interim

        try:
            logging.debug('start waiting', extra={"obj_id": self.agent.name})
            await asyncio.wait_for(self.round_finished.wait(), timeout=30)
        except TimeoutError:
            print('wait for timeout')
            logging.debug(f'Order ID:{proposals_to_update[0].call_for_proposal.order.identification} await timeout',extra={"obj_id": self.agent.name})
            print(self.agent.name, "TimeoutError",
                  len(self.proposals_to_accept),
                  len(self.requests_shelved)) #Agent model sheet: process : material_part_unloading_gear_shift_brakes_etn :function call war 2x gleich, aber nicht die lösung
            print(list(self.proposals_to_accept.keys()))
            for i, pta in self.proposals_to_accept.items():
                print(pta.call_for_proposal.client_object.process.name) #VAP:brake disc rim brakes standard is True,
            self.proposals_evaluated = {}
            self.round_finished.set()
            return
        self.proposals_evaluated = {}
        self.finishing_opened.clear()
        logging.debug(f'Order ID:{proposals_to_update[0].call_for_proposal.order.identification} finished negotiation round',
                      extra={"obj_id": self.agent.name})

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Finished negotiation round")

    async def get_round_finished_status(self):
        try:
            await self.round_finished.wait()
        except TimeoutError:
            print("round_finished status request timed out")
        self.round_finished = None

        return True

    async def _schedule(self, requests_met):
        """Trigger the scheduling process for processes be possible on the resources administrated
        Firstly it is checked if one or more processes be possible to schedule on the resource
        If only one process is scheduled on the resource it did not need to be scheduled from a coordinator
        (that take also other schedule shares from other agents)"""

        # if not requests_met:
        #     await self._set_scheduling_finished([], [], [])
        #     return [], [], []
        # print(get_debug_str(self.agent.name, self.__class__.__name__), "Start")
        # print(get_debug_str(self.agent.name, self.__class__.__name__), "Preparing scheduling")

        proposals_to_schedule, proposals_to_reject = _get_proposals_to_schedule(requests_met)
        lowest_level_elements, higher_level_elements = _get_lowest_level_elements(proposals_to_schedule)

        # print("Parents to schedule",
        #       set([(proposal.process_executions_components_parent.node_identification,
        #             proposal.process_executions_components_parent.goal_item.identification)
        #            for proposal in proposals_to_schedule]))

        # print(get_debug_str(self.agent.name, self.__class__.__name__), "Start scheduling")
        alternatives_to_refuse, alternatives_to_update, alternatives_new = \
            await self._coordinate_scheduling_elements(lowest_level_elements)

        if alternatives_new:
            print("new alternatives")
            raise Exception

        proposals_new = self._create_proposals_coordination(alternatives_new)

        # translation to proposals
        alternatives_to_refuse += proposals_to_reject  # Maybe it is possible to do it before
        proposals_to_schedule_set = set(proposals_to_schedule)
        proposals_to_refuse, additional_components_to_refuse = \
            _get_proposals_by_process_executions_components(proposals_to_schedule_set,
                                                            alternatives=alternatives_to_refuse)
        proposals_to_update, additional_components_to_update = \
            _get_proposals_by_process_executions_components(proposals_to_schedule_set,
                                                            alternatives=alternatives_to_update)

        # print(self.agent.name, len(additional_components_to_refuse))
        # additional_components are usually the connectors
        for additional_component_to_refuse in additional_components_to_refuse:
            additional_component_to_refuse.reject()
            if additional_component_to_refuse.process_executions_components_parent:
                additional_component_to_refuse.process_executions_components_parent.reject()

        for additional_component_to_update in additional_components_to_update:
            additional_component_to_update.accept()
            if additional_component_to_update.process_executions_components_parent:
                additional_component_to_update.process_executions_components_parent.accept()

        # print("Parents to update",
        #               [(proposal.node_identification,
        #                 proposal.process_executions_components_parent.node_identification,
        #                 proposal.process_executions_components_parent.goal_item.identification,
        #                 proposal.process_executions_components_parent.cfp_path)
        #                for proposal in proposals_to_update + additional_components_to_update
        #                if proposal.process_executions_components_parent])
        #
        # print("Parents to refuse",
        #       [(proposal.node_identification,
        #         proposal.process_executions_components_parent.node_identification,
        #         proposal.process_executions_components_parent.goal_item.identification,
        #         proposal.process_executions_components_parent.cfp_path)
        #        for proposal in proposals_to_refuse + additional_components_to_refuse
        #        if proposal.process_executions_components_parent])

        for proposal in proposals_to_refuse:
            proposal.reject()

        # for proposal in proposals_to_update:
        #     proposal.accept()

        # for proposal in proposals_to_update:  # ToDo: check if necessary
        #     sub_proposals = proposal.get_sub_proposals()
        #     if not sub_proposals:
        #         continue
        #
        #     while sub_proposals:
        #         new_sub_proposals = []
        #         for sub_proposal in sub_proposals:
        #             sub_proposal.accept()
        #             new_sub_proposals.extend(sub_proposal.get_sub_proposals())
        #
        #         sub_proposals = new_sub_proposals.copy()
        #
        # for proposal in proposals_to_refuse:
        #     sub_proposals = proposal.get_sub_proposals()
        #     if not sub_proposals:
        #         continue
        #
        #     while sub_proposals:
        #         new_sub_proposals = []
        #         for sub_proposal in sub_proposals:
        #             sub_proposal.reject()
        #             new_sub_proposals.extend(sub_proposal.get_sub_proposals())
        #
        #         sub_proposals = new_sub_proposals.copy()

        proposals_to_refuse += proposals_to_reject  # Maybe it is possible to do it before
        # self.remove_open_proposals(proposals_to_refuse + proposals_to_update)

        return proposals_to_refuse, proposals_to_update

    async def _coordinate_scheduling_elements(self, process_execution_elements_to_schedule):
        """Send a coordination request to the scheduler if needed"""

        await self._send_coordination_request(process_execution_elements_to_schedule)

        alternatives_to_refuse, alternatives_to_update, alternatives_new = await self._wait_for_coordination_response()

        return alternatives_to_refuse, alternatives_to_update, alternatives_new

    async def _send_coordination_request(self, process_executions_components):
        """Send the process_executions_components to a central coordination institution"""

        if "coordinator_agent" not in self.agent.address_book:
            raise Exception("coordinator_agent not in address_book")

        coordinator_name = self.agent.address_book["coordinator_agent"]

        msg_content = CoordinationCO(process_executions_components=process_executions_components,
                                     resources_preferences=self.agent.preferences)

        # object not json serializable
        await self.agent.send_msg(behaviour=self, receiver_list=[coordinator_name], msg_body=msg_content,
                                  message_metadata={"performative": "request",
                                                    "ontology": "COORDINATION",
                                                    "language": "OWL-S"})

    async def _react_on_coordination_response(self, process_executions_components_d: dict):
        """Receive a message from the coordinator agent and pass them to another method
        (_wait_for_coordination_response) though set_result"""


        self._coordination_result_object = process_executions_components_d
        # print(self.agent.name, "set")
        self._coordination_finished.set()

    async def _wait_for_coordination_response(self):
        """Wait for the coordination response"""
        self._coordination_finished.clear()
        await self._coordination_finished.wait()
        # print(self.agent.name, "ready")
        alternatives_to_refuse =  self._coordination_result_object["REFUSED"]
        alternatives_to_update =  self._coordination_result_object["UPDATED"]
        alternatives_new =  self._coordination_result_object["NEW"]

        self._coordination_result_object = None

        return alternatives_to_refuse, alternatives_to_update, alternatives_new

    async def _send_refinement_updates(self, proposals_to_refuse, proposals_to_update):
        """Send the refinement updates to the initiator"""

        if proposals_to_refuse:
            await self._send_refinement_update(proposals=proposals_to_refuse,
                                               performative="refuse-proposal",
                                               ontology="PLANNING_PROPOSAL")
            self.remove_open_proposals(proposals_to_refuse)

        if proposals_to_update:
            await self._send_refinement_update(proposals=proposals_to_update,
                                               performative="update-proposal",
                                               ontology="PLANNING_PROPOSAL")
            self.proposals_to_accept |= {proposal.identification: proposal
                                         for proposal in proposals_to_update}
            self.remove_open_proposals(proposals_to_update)

        # print("Other refinment updates", len(set(proposals_to_refuse + proposals_to_update)),
        #       [proposal.call_for_proposal.reference_cfp.identification
        #        if proposal.call_for_proposal.reference_cfp
        #        else None
        #        for proposal in proposals_to_refuse + proposals_to_update])
        self.remove_proposals_request_met(proposals_to_refuse + proposals_to_update)

    async def _send_refinement_update(self, proposals, performative, ontology):

        # summary per sender
        sender_name_proposals = {}
        for proposal in proposals:
            sender_name_proposals.setdefault(proposal.call_for_proposal.sender_name, []).append(proposal)

        # send
        for receiver, proposals in sender_name_proposals.items():
            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       f" Proposal IDs ({performative}): ", [p.identification for p in proposals])

            proposals_communication_object = ListCO(proposals)
            # object not json serializable
            await self.agent.send_msg(behaviour=self, receiver_list=[receiver], msg_body=proposals_communication_object,
                                      message_metadata={"performative": performative,
                                                        "ontology": ontology,
                                                        "language": "OWL-S"})

    def remove_proposals_request_met(self, proposals):
        """Remove the proposal from request met because of two possible reasons
        The proposals are passed to the refinement, or they are rejected"""

        requests_to_remove = []
        for cfp, request_proposals in self.requests_met.items():
            request_proposals = [proposal for proposal in request_proposals if proposal not in proposals]
            self.requests_met[cfp] = request_proposals

            if not request_proposals:
                requests_to_remove.append(cfp)

        for cfp in requests_to_remove:
            del self.requests_met[cfp]

    async def update_response(self, cfp, proposals, type_):
        """
        method only for passing to the initiator behaviour
        Normally used from the scheduler (or even though from the planning phase) back to the first call for proposal
        """

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Proposal to {type_}: ",
        #       [proposal.identification for proposal in proposals])

        # if type_ == "REFUSE":
        #     await self._refuse_sub_proposals(proposals)

        negotiation_object_refinement_needed, shelved = self._get_negotiation_object_refinement_needed(cfp)
        if negotiation_object_refinement_needed is None:
            return

        new_proposals_evaluated = {proposal: type_ for proposal in proposals}
        self.proposals_evaluated.update(new_proposals_evaluated)

        # if all sub_proposals in proposals_evaluated merge them
        sub_proposals = set(sub_proposal
                            for proposal in negotiation_object_refinement_needed
                            for sub_proposal in proposal.get_sub_proposals())
        proposals_available = set(self.proposals_evaluated.keys())
        all_proposals_available = sub_proposals.issubset(proposals_available)

        if not all_proposals_available:
            return

        proposals_to_refuse, proposals_to_update = \
            self._get_proposals_to_response(negotiation_object_refinement_needed, type_)

        if not shelved:
            # print("Proposals", cfp.identification, len(set(proposals)),
            #       [proposal.call_for_proposal.reference_cfp.identification
            #        if proposal.call_for_proposal.reference_cfp
            #        else None
            #        for proposal in proposals_to_refuse + proposals_to_update])
            await self._send_refinement_updates(proposals_to_refuse, proposals_to_update)

        else:
            # remove the proposals from self.requests_shelved or self.requests_met
            for proposal_to_refuse in proposals_to_refuse:
                # print(f"Remove proposal with ID: {proposal_to_refuse.identification}")
                if proposal_to_refuse in negotiation_object_refinement_needed:
                    negotiation_object_refinement_needed.remove(proposal_to_refuse)

        # print(self.agent.name, self.requests_met, self.requests_shelved)
        if cfp in self.requests_shelved:  # from shelved to ready_for_proposal
            request_proposals_to_remove = self.requests_shelved[cfp]

            if request_proposals_to_remove:

                # ToDo: testing in bigger scenarios
                # workaround if the planning process failed
                proposals_to_reject = [proposal
                                       for proposal in request_proposals_to_remove
                                       if not proposal.acceptable()]

                if proposals_to_reject:
                    self.requests_shelved[cfp] = \
                        list(set(request_proposals_to_remove).difference(set(proposals_to_reject)))
                    # print("Case - not possible")
                    # await self._refuse_sub_proposals(proposals_to_reject)
                    if request_proposals_to_remove:
                        self.requests_ready_for_refuse[cfp] = self.requests_shelved.pop(cfp)

                    else:
                        self.requests_ready_for_proposal[cfp] = self.requests_shelved.pop(cfp)

                else:
                    self.requests_ready_for_proposal[cfp] = self.requests_shelved.pop(cfp)

            else:
                self.requests_ready_for_refuse[cfp] = self.requests_shelved.pop(cfp)
                if proposals_to_refuse:
                    await self._refuse_sub_proposals(proposals_to_refuse)

                # if len(request_shelved["proposals"]) == 1:
                #     requests_to_remove.append(request_shelved)
                #     self.requests_ready_for_proposal.append(request_shelved)

        if not self.requests_shelved and not self.requests_met:
            if not self._ready_for_proposal_creation.is_set():
                self._ready_for_proposal_creation.set()

    def _get_negotiation_object_refinement_needed(self, cfp):
        """Determine if the cfp and the respective proposals can be found in met or shelved"""
        shelved = False
        if cfp is None:
            negotiation_object_refinement_needed = None
            return negotiation_object_refinement_needed, shelved

        if cfp in self.requests_met:
            negotiation_object_refinement_needed = self.requests_met[cfp]
            shelved = False

        elif cfp in self.requests_shelved:
            negotiation_object_refinement_needed = self.requests_shelved[cfp]

        else:
            negotiation_object_refinement_needed = None

        return negotiation_object_refinement_needed, shelved

    async def _refuse_sub_proposals(self, proposals):
        # the way down

        proposals_to_refuse_down = {}
        for proposal in proposals:
            proposal.reject()

            for sub_proposal in proposal.get_sub_proposals():
                if sub_proposal not in self.proposals_evaluated:
                    # check if all parents rejected
                    sub_proposal.reject()
                    proposals_to_refuse_down.setdefault(sub_proposal.provider, []).append(sub_proposal)
                elif self.proposals_evaluated[sub_proposal] != "REFUSE":
                    sub_proposal.reject()
                    proposals_to_refuse_down.setdefault(sub_proposal.provider, []).append(sub_proposal)

        if proposals_to_refuse_down:
            await self.connection_behaviour.reject_proposals_provider(proposals_to_refuse_down)

    def _get_proposals_to_response(self, proposals, type_):
        # update if necessary
        proposals_to_refuse = []
        proposals_to_update = []

        if type_ == "REFUSE":
            proposals_to_refuse = proposals
            for proposal in proposals:
                proposal.reject()

        elif type_ == "UPDATE":
            proposals_to_update = proposals

        #
        # # the way up
        # for proposal in proposals:
        #     types = [self.proposals_evaluated[sub_proposal]
        #              for sub_proposal in proposal.get_sub_proposals()]
        #
        #     if "REFUSE" in types:
        #
        #         proposals_to_refuse.append(proposal)
        #
        #     else:
        #         # try:
        #         #     proposal.update()
        #         #
        #         # except:
        #         #     print("Case")
        #         #     proposals_to_refuse.append(proposal)
        #
        #         proposals_to_update.append(proposal)

        return proposals_to_refuse, proposals_to_update

    async def ready_for_proposal_creation(self):
        """ready_for_proposal_creation means that the planning phase is already finished and
        the proposals can be created (right?)"""
        self._ready_for_proposal_creation = asyncio.Event()
        await self._ready_for_proposal_creation.wait()

        # it = 0
        #
        # while True:
        #     if self._ready_for_proposal_creation is not None:
        #         if self._ready_for_proposal_creation.done():
        #             break
        #     else:
        #         break
        #
        #     it += 1
        #     await asyncio.sleep(1)
        #     if it > 5:
        #         print

    async def _process_shelved_proposals(self):
        """The possible_proposals build the basis for the shelved_proposals"""
        open_proposals_to_add = []
        for negotiation_object, proposals in self.requests_ready_for_proposal.items():
            # the proposal is sent to the requester
            # print("Proposals", negotiation_object.sender_name, len(proposals))
            # print("Accept:", negotiation_object.sender_name)
            await self._send_proposals(sender=negotiation_object.sender_name,
                                       proposals=proposals)
            open_proposals_to_add.extend(proposals)

        self.add_open_proposals(open_proposals_to_add)

        for negotiation_object, proposals in self.requests_ready_for_refuse.items():
            # print("Reject:", negotiation_object.sender_name, len(proposals))
            if len(proposals) > 0:
                print("Exception")
                raise Exception
            # print("Refuse:", negotiation_object.sender_name)
            await self._send_refuse(sender=negotiation_object.sender_name,
                                    negotiation_object=negotiation_object)

        self.requests_ready_for_proposal = {}
        self.requests_ready_for_refuse = {}

    async def _send_process_executions(self, proposals_process_executions_creation):
        """Send accepted_proposals to the requester"""

        receiver_process_executions = {}
        for proposal in proposals_process_executions_creation:
            # print("Accepted",
            #       proposal.goal_item.process.name,
            #       [resource.name for resource in proposal.goal_item.get_resources()])

            process_executions = proposal.get_process_executions(type_="ACCEPTED")
            receiver_process_executions.setdefault(proposal.call_for_proposal.sender_name,
                                                   []).extend(process_executions)

        for receiver, process_executions in receiver_process_executions.items():
            process_executions_co = ListCO(process_executions)
            msg_content = process_executions_co

            await self.agent.send_msg(behaviour=self, receiver_list=[receiver], msg_body=msg_content,
                                      message_metadata={"performative": "inform",
                                                        "ontology": "PROPOSAL",
                                                        "language": "OWL-S"})

    def add_open_proposals(self, new_requests_to_process):
        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Proposals added")
        new_open_proposals = {proposal.identification: proposal
                              for proposal in new_requests_to_process}

        self.open_proposals |= new_open_proposals

    def remove_open_proposals(self, open_proposals_to_remove, end=False):

        for open_proposal_to_remove in open_proposals_to_remove:
            if open_proposal_to_remove.identification in self.open_proposals:
                del self.open_proposals[open_proposal_to_remove.identification]

        # open_proposals = [(id_, open_proposal.call_for_proposal.identification)
        #                   for id_, open_proposal in self.open_proposals.items()]
        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Open proposals {open_proposals}")

        if not end:
            return

    def get_results(self) -> [Dict, Dict]:
        """
        A callback function that provides the results of the negotiation.
        :return accepted_proposals: a list of requester_agent_name: the name of the agent that planning
        the process_executions_components and process_executions_components: the specified process_executions_components
        from the accepted_proposal
        if a proposal is accepted - else negotiation failed
        """
        # take the elements from the list
        accepted_proposals_lst, self.accepted_proposals = self.accepted_proposals.copy(), []
        rejected_proposals_lst, self.rejected_proposals = self.rejected_proposals.copy(), []

        accepted_proposals = {}
        proposal=None
        for requester_agent_jid, proposal in accepted_proposals_lst:
            if (isinstance(proposal.call_for_proposal, ResourceCallForProposal) or
                    isinstance(proposal.call_for_proposal, PartCallForProposal)):
                if isinstance(requester_agent_jid, str):
                    provider_agent_name = requester_agent_jid
                else:
                    provider_agent_name = requester_agent_jid.localpart
                accepted_process_executions = proposal.get_participating_process_executions(type_="ACCEPTED")
                accepted_proposals.setdefault(provider_agent_name,
                                              []).extend(accepted_process_executions)
        if proposal is not None:
            order_id = proposal.call_for_proposal.order.identification
        else:
            order_id = 0

        logging.debug(f'Order ID {order_id} get_result 0.1', extra={"obj_id": self.agent.name})
        # accepted_process_executions = [elem for lst in list(accepted_proposals.values()) for elem in lst]

        # print(self.agent.name, "Accepted",
        #       [(process_execution.identification, process_execution.process.name)
        #        for process_execution in accepted_process_executions])

        self._handle_reservations(accepted_proposals_lst)
        logging.debug(f'Order ID {order_id} get_result 0.2', extra={"obj_id": self.agent.name})

        rejected_proposals = {}
        for requester_agent_jid, proposal in rejected_proposals_lst:
            if (isinstance(proposal.call_for_proposal, ResourceCallForProposal) or
                    isinstance(proposal.call_for_proposal, PartCallForProposal)):
                if isinstance(requester_agent_jid, str):
                    provider_agent_name = requester_agent_jid
                else:
                    provider_agent_name = requester_agent_jid.localpart

                rejected_process_executions = \
                    [process_execution
                     for process_execution in proposal.get_participating_process_executions(type_="REJECTED")]
                rejected_proposals.setdefault(provider_agent_name,
                                              []).extend(rejected_process_executions)
        logging.debug(f'Order ID {order_id} get_result 0.3', extra={"obj_id": self.agent.name})

        if not (accepted_proposals or rejected_proposals):
            logging.debug(f'Order ID {order_id} get_result 0.4',
                          extra={"obj_id": self.agent.name})
            return accepted_proposals, rejected_proposals
        if self.agent.new_process_executions_available is not None:
            if not self.agent.new_process_executions_available.is_set():
                self.agent.new_process_executions_available.set()
        logging.debug(f'Order ID {order_id} get_result 0.5', extra={"obj_id": self.agent.name})
        return accepted_proposals, rejected_proposals
