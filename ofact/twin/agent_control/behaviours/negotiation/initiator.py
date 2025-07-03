"""
This file encapsulates the initiator of the negotiation.
http://www.fipa.org/specs/fipa00037/SC00037J.html#_Toc26729693
"""
# Imports Part 1: Standard Imports
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour
from ofact.twin.agent_control.behaviours.negotiation.objects import Proposal, CallForProposal
from ofact.twin.agent_control.helpers.communication_objects import ListCO

if TYPE_CHECKING:
    from ofact.twin.agent_control.behaviours.negotiation.CNET import CNETNegotiationBehaviour
# Module-Specific Constants
logger = logging.getLogger("CNETInitiator")


class InitiatorBehaviour(DigitalTwinCyclicBehaviour):
    """
    The initiator is used to initiate call for proposals (e.g. Process (Organization) Request).
    With accordance to the CNET, the acceptance and rejection of a proposal is done too.
    Furthermore, the proposals that are refused or proposed, are collected and send to the participant on the way back.
    """

    planning_proposal_template = {"metadata": {"performative": "propose",
                                               "ontology": "PLANNING_PROPOSAL",
                                               "language": "OWL-S"}}
    planning_refuse_template = {"metadata": {"performative": "refuse",
                                             "ontology": "PLANNING_PROPOSAL",
                                             "language": "OWL-S"}}
    planning_proposal_refuse_template = {"metadata": {"performative": "refuse-proposal",
                                                      "ontology": "PLANNING_PROPOSAL",
                                                      "language": "OWL-S"}}
    planning_proposal_update_template = {"metadata": {"performative": "update-proposal",
                                                      "ontology": "PLANNING_PROPOSAL",
                                                      "language": "OWL-S"}}

    templates = [planning_proposal_template, planning_refuse_template, planning_proposal_refuse_template,
                 planning_proposal_update_template]

    def __init__(self, negotiation_time_limit, connection_behaviour):
        """
        The CNETRequester start the negotiation process with the call_for_proposal, wait for the responses/ proposals,
        evaluate them based of his own preferences and accept the best one. All others are rejected.
        :param negotiation_time_limit: a timestamp (also seconds possible) that determine the deadline for the proposals
        come back from the responders
        """
        super(InitiatorBehaviour, self).__init__()

        self.negotiation_time_limit = negotiation_time_limit  # timestamp
        self.connection_behaviour: CNETNegotiationBehaviour = connection_behaviour

        self.metadata_conditions_lst = [{"performative": "propose", "ontology": "PLANNING_PROPOSAL"},
                                        {"performative": "refuse", "ontology": "PLANNING_PROPOSAL"},
                                        {"performative": "refuse-proposal", "ontology": "PLANNING_PROPOSAL"},
                                        {"performative": "update-proposal", "ontology": "PLANNING_PROPOSAL"}]

        self.message_handling_func_mapper = {"propose": self._receive_propose_message,
                                             "refuse": self._receive_refuse_message,
                                             "refuse-proposal": self._receive_refuse_proposal_message,
                                             "update-proposal": self._receive_update_proposal_message}

        # agent_name/_jid in the proposal object
        self.received_proposals: dict[CallForProposal, list[Proposal]] = {}
        self.refused_call_for_proposals: dict[CallForProposal, list[Proposal]] = {}

        self.planning_proposals_waiting_for_inform = {}
        self.planning_proposals_waiting_for_update = {}
        self.planning_proposals_waiting_for_refuse = {}

        self._accepted_proposals: list[tuple[str, dict]] = []  # agent_name/_jid and the proposal object as dict
        self._rejected_proposals: list[tuple[str, dict]] = []  # agent_name/_jid and the proposal object as dict

        self.process_executions_evaluated = None

        self.STARTED = False

    @classmethod
    def get_templates(cls):
        return deepcopy(cls.templates)

    async def call_for_proposal(self, call_for_proposals, providers):
        """The call for proposal sends the negotiation_object to provider agents. They should prepare an offer."""
        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" CfP with ID: '{[call_for_proposal.identification for call_for_proposal in call_for_proposals]}' "
        #       f"sent to providers {providers}")

        await self._send_negotiation_object(negotiation_objects=call_for_proposals,
                                            providers=providers,
                                            performative="cfp",
                                            ontology="CFP")

        # print(get_debug_str(self.agent.name, self.__class__.__name__) + f" Cfp sent", providers)

    async def accept_proposals_provider(self, accepted_proposals_provider):
        """The proposal (to propose) is sent to the proposal creator with specified information and a price.
        He should evaluate if he can achieve the specified proposal."""

        for provider, accepted_proposals_lst in accepted_proposals_provider.items():
            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       f" Send accepted proposal: "
            #       f"'{[accepted_proposal.identification for accepted_proposal in accepted_proposals_lst]}'")

            await self._send_negotiation_object(negotiation_objects=accepted_proposals_lst,
                                                providers=[provider],
                                                performative="accept-proposal",
                                                ontology="PLANNING_PROPOSAL")

        # logger.debug(get_debug_str(self.agent.name, self.__class__.__name__) + f" Propose planning proposal sent")

    async def reject_proposals_provider(self, rejected_proposals_provider):
        """The refuse of a planned_proposal"""

        for provider, rejected_proposals_lst in rejected_proposals_provider.items():
            # print(get_debug_str(self.agent.name, self.__class__.__name__) +
            #       f" Send rejected proposal with IDs: "
            #       f"'{[rejected_proposal.identification for rejected_proposal in rejected_proposals_lst]}'")

            await self._send_negotiation_object(negotiation_objects=rejected_proposals_lst,
                                                providers=[provider],
                                                performative="reject-proposal",
                                                ontology="PLANNING_PROPOSAL")

        # logger.debug(get_debug_str(self.agent.name, self.__class__.__name__) + f" Refuse planning proposal sent")

    async def _send_negotiation_object(self, negotiation_objects, providers, performative, ontology):
        # store the negotiation_object in the agent_model
        list_communication_object = ListCO(negotiation_objects)
        msg_content = list_communication_object

        await self.agent.send_msg(behaviour=self, receiver_list=providers, msg_body=msg_content,
                                  message_metadata={"performative": performative,
                                                    "ontology": ontology,
                                                    "language": "OWL-S"})

    async def run(self):
        await super().run()

        msg_received = await self.agent.receive_msg(self, timeout=10,
                                                    metadata_conditions_lst=self.metadata_conditions_lst)
        if msg_received is None:
            return
        msg_content, msg_sender, msg_ontology, msg_performative = msg_received
        # create a task to process the message impact asynchronously
        proposals = msg_content

        task = asyncio.create_task(self.message_handling_func_mapper[msg_performative](proposals, msg_sender))

    async def _receive_propose_message(self, proposals, sender):
        """ Receive the proposals from the responder and write them into the received_proposals list"""
        if not proposals:
            raise Exception

        for proposal in proposals:
            self.received_proposals.setdefault(proposal.call_for_proposal.reference_cfp, []).append(proposal)

        # print(get_debug_str(self.agent.name, self.__class__.__name__) +
        #       f" Proposals {[proposal.identification for proposal in proposals]} available")

        self.connection_behaviour.store_proposal(proposals, provider=sender)

        return bool(proposals)

    async def _receive_refuse_message(self, call_for_proposal_refused, sender):
        """Receive a refuse message"""

        if isinstance(call_for_proposal_refused, CallForProposal):
            self.connection_behaviour.record_call_for_proposal_refuse(call_for_proposal_refused, provider=sender)

        return bool(call_for_proposal_refused)

    async def _receive_refuse_proposal_message(self, refused_proposals, _):
        """Receive a refuse message -
        these messages are needed to determine if it should be waited on possible proposals"""
        await self._set_on_waiting_list(proposals=refused_proposals,
                                        waiting_list=self.planning_proposals_waiting_for_refuse)

    async def _receive_update_proposal_message(self, updated_proposals, _):
        """unchanged"""
        await self._set_on_waiting_list(proposals=updated_proposals,
                                        waiting_list=self.planning_proposals_waiting_for_update)

    async def _set_on_waiting_list(self, proposals, waiting_list):
        """Ensure that all proposals of a call for proposal are evaluated before the proposals are forwarded
        for further processing"""

        # normally the method is used to come from the leave nodes and reach the root_nodes
        # because the refuse of not possible sub_trees is necessary , they are done in parallel
        # (they are not considered in a rejection from the highest level)

        # sub_proposals_to_refuse_provider = {}
        for proposal in proposals:
            reference_cfp = proposal.call_for_proposal.reference_cfp
            waiting_list.setdefault(reference_cfp,
                                    []).append(proposal)

            # proposition: if all proposals of a call for proposal are refused, the call for proposal is refused
            if reference_cfp not in self.received_proposals:
                continue

            if len(self.received_proposals[reference_cfp]) > 0:
                # simply delete the proposal
                if proposal in self.received_proposals[reference_cfp]:
                    self.received_proposals[reference_cfp].remove(proposal)

                # print(get_debug_str(self.agent.name, self.__class__.__name__), reference_cfp.identification,
                #       len(self.received_proposals[reference_cfp]))
            #
            # if reference_cfp:
            #     print(self.agent.name, reference_cfp.identification, reference_cfp.__class__.__name__,
            #       len(self.received_proposals[reference_cfp]))
            # else:
            #     print(self.agent.name, reference_cfp,  len(self.received_proposals[reference_cfp]))

            if len(self.received_proposals[reference_cfp]) == 0:
                del self.received_proposals[reference_cfp]
                await self._assess_consequences(reference_cfp)

    async def _assess_consequences(self, reference_cfp):
        """assumption/ condition: if this method is called, all proposals related to the reference_cfp are
        already available"""

        if reference_cfp in self.planning_proposals_waiting_for_refuse:
            waiting_for_refuse = self.planning_proposals_waiting_for_refuse.pop(reference_cfp)
            await self.connection_behaviour.trigger_refinement_update(reference_cfp, waiting_for_refuse, "REFUSE")

        if reference_cfp in self.planning_proposals_waiting_for_update:
            waiting_for_update = self.planning_proposals_waiting_for_update.pop(reference_cfp)
            await self.connection_behaviour.trigger_refinement_update(reference_cfp, waiting_for_update, "UPDATE")
