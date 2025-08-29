"""
This file encapsulates the negotiators. A negotiator is responsible to negotiate on behalf of an agent to fulfill his
interests.
http://www.fipa.org/specs/fipa00037/SC00037J.html#_Toc26729693
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import asyncio
import logging
from abc import ABCMeta
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.basic import \
    DigitalTwinCyclicBehaviour  # ToDo: maybe also OneShotBehaviour possible/ better
from ofact.twin.agent_control.behaviours.negotiation.initiator import InitiatorBehaviour
from ofact.twin.agent_control.behaviours.negotiation.participant import ParticipantBehaviour

if TYPE_CHECKING:
    from ofact.twin.agent_control.behaviours.negotiation.objects import CallForProposal, Proposal

# Module-Specific Constants
logger = logging.getLogger("CNET")


class CNETNegotiationBehaviour(DigitalTwinCyclicBehaviour, metaclass=ABCMeta):
    """Interface/ manager between the initiator and the participant behaviour"""

    def __init__(self, negotiation_time_limit, collection_phase_period: float, collection_volume_limit: int):
        super(CNETNegotiationBehaviour, self).__init__()
        self.behaviours = {}

        self.negotiation_completed = None
        self.process_executions_evaluated = None

        self.behaviour_templates = [(InitiatorBehaviour, {"connection_behaviour": self,
                                                          "negotiation_time_limit": negotiation_time_limit}),
                                    (ParticipantBehaviour, {"connection_behaviour": self,
                                                            "collection_phase_period": collection_phase_period,
                                                            "collection_volume_limit": collection_volume_limit})]

        self.InitiatorBehaviour = None
        self.ParticipantBehaviour = None

        self.response_subscriptions_providers = {}
        self.response_subscriptions_callback = {}
        self.proposals = {}

    def get_behaviour_templates(self) -> list[dict]:
        """called initially to set the behaviours"""

        behaviour_with_templates = []
        for behaviour_class, input_param_dict in self.behaviour_templates:
            templates = behaviour_class.get_templates()

            behaviour = behaviour_class(**input_param_dict)

            if InitiatorBehaviour.__name__ == behaviour.__class__.__name__:
                self.InitiatorBehaviour = behaviour
                self.behaviours[InitiatorBehaviour] = behaviour

            elif ParticipantBehaviour.__name__ in behaviour.__class__.__name__:
                self.ParticipantBehaviour = behaviour
                self.behaviours[ParticipantBehaviour] = behaviour

            behaviour_with_templates.append({"behaviour": behaviour,
                                             "templates": templates})

        return behaviour_with_templates

    def convert_providers(self, providers):
        """Find digital twin objects or strings in the address and return the respective agent jid"""
        if not providers:
            return []
        if isinstance(providers[0], str):
            return providers

        providers = list(set([self._get_providers(provider)
                              for provider in providers]))
        return providers

    def _get_providers(self, provider):
        if provider in self.InitiatorBehaviour.agent.address_book:
            return self.InitiatorBehaviour.agent.address_book[provider]
        else:
            return self.InitiatorBehaviour.agent.address_book[provider.entity_type]

    def transport_call_for_proposal(self, cfp, providers):
        if not isinstance(providers[0], str):  # sample inspection
            providers = self.convert_providers(providers)

        self.response_subscriptions_providers.setdefault(cfp.identification,
                                                         []).extend(providers)
        if cfp.identification not in self.response_subscriptions_callback:
            self.response_subscriptions_callback[cfp.identification] = asyncio.get_event_loop().create_future()

    async def call_for_proposal(self, call_for_proposals: CallForProposal | list[CallForProposal], providers):
        """Initiate call for proposal"""

        if not isinstance(call_for_proposals, list):
            call_for_proposals = [call_for_proposals]
        if not isinstance(providers[0], str):  # sample inspection
            providers = self.convert_providers(providers)

        for call_for_proposal in call_for_proposals:
            self.response_subscriptions_providers.setdefault(call_for_proposal.identification, []).extend(providers)
            if call_for_proposal.identification not in self.response_subscriptions_callback:
                self.response_subscriptions_callback[call_for_proposal.identification] = \
                    asyncio.get_event_loop().create_future()

        await self.InitiatorBehaviour.call_for_proposal(call_for_proposals, providers)

    async def await_callback(self, call_for_proposal_identifications, behaviour_name="OrderManagement"):
        """Wait for a response of a call for proposal or a proposal made"""

        if not call_for_proposal_identifications:
            return []
        # print(get_debug_str("CNET", "") + f" Wait on: {call_for_proposal_identifications} {behaviour_name}")
        print(f'call fpr proposal identifications {call_for_proposal_identifications}')
        callbacks_to_await = [self.response_subscriptions_callback[call_for_proposal_id]
                              for call_for_proposal_id in call_for_proposal_identifications]
        print(f' callbacks: {callbacks_to_await}')
        try:
            await asyncio.wait(callbacks_to_await, timeout=20)
        except asyncio.TimeoutError:
            print('CNET Timeout error')
        proposals = []
        print(f'proposals: {self.proposals}')
        for call_for_proposal_identification in call_for_proposal_identifications:
            if call_for_proposal_identification in self.proposals:  # if proposals available
                proposals += self.proposals[call_for_proposal_identification]
                del self.proposals[call_for_proposal_identification]

            del self.response_subscriptions_providers[call_for_proposal_identification]
            del self.response_subscriptions_callback[call_for_proposal_identification]
        print(f'len proposals: {len(proposals)}')
        return proposals

    def store_proposal(self, proposals: list[Proposal], provider):
        """Store the proposal means also to record that the proposal is there for further processing"""

        if not proposals:
            return

        if type(provider) != str:  # unpack the jid
            provider = provider.localpart

        cfp_ids = []
        for proposal in proposals:
            # print(f"Proposal with ID '{proposal.identification}' stored")
            call_for_proposal_id = proposal.call_for_proposal.identification
            self.proposals.setdefault(call_for_proposal_id,
                                      []).append((proposal, provider))
            cfp_ids.append(call_for_proposal_id)

        for cfp_id in set(cfp_ids):
            self._set_call_for_proposal_called(cfp_id, provider)

    def record_call_for_proposal_refuse(self, call_for_proposal: CallForProposal, provider):
        """Record call for proposal as refused (provider will deliver no proposal)"""

        if type(provider) != str:  # unpack the jid
            provider = provider.localpart
        self._set_call_for_proposal_called(call_for_proposal.identification, provider)

    def _set_call_for_proposal_called(self, call_for_proposal_id: int, provider):
        """Set the call for proposal as called"""

        # record provider as 'has delivered'
        # (if more than one proposal - maybe the provider is already removed from list)
        if call_for_proposal_id not in self.response_subscriptions_providers:
            return

        if provider in self.response_subscriptions_providers[call_for_proposal_id]:
            self.response_subscriptions_providers[call_for_proposal_id].remove(provider)

        if not self.response_subscriptions_providers[call_for_proposal_id]:
            try:
                self.response_subscriptions_callback[call_for_proposal_id].set_result(True)
            except asyncio.exceptions.InvalidStateError:
                print(f"InvalidStateError: {call_for_proposal_id}")

    async def accept_proposal(self, proposals):
        """Accept a proposal - Mark the proposal as accepted and send a notice"""

        accepted_proposals_provider = {}
        for proposal in proposals:
            accepted_proposals_provider.setdefault(proposal.provider,
                                                   []).append(proposal)
            proposal.accept()

        await self.accept_proposals_provider(accepted_proposals_provider)

    async def reject_proposal(self, proposals):
        """Reject a proposal - Mark the proposal as reject and send a notice"""

        rejected_proposals_provider = {}
        for proposal in proposals:
            rejected_proposals_provider.setdefault(proposal.provider,
                                                   []).append(proposal)
            proposal.reject()

        await self.reject_proposals_provider(rejected_proposals_provider)

    async def accept_proposals_provider(self, accepted_proposals_provider):
        return await self.InitiatorBehaviour.accept_proposals_provider(accepted_proposals_provider)

    async def reject_proposals_provider(self, rejected_proposals_provider):
        return await self.InitiatorBehaviour.reject_proposals_provider(rejected_proposals_provider)

    #### participant ####

    async def trigger_refinement_update(self, cfp, proposals, type_):
        """Done from the initiator on the way back to the order"""

        await self.ParticipantBehaviour.update_response(cfp, proposals, type_)

    def negotiation_round_finished_info(self):
        """Check if the negotiation round is finished"""

        if self.ParticipantBehaviour:
            return self.ParticipantBehaviour.round_finished.is_set()
        else:
            return False

    def get_results(self):
        """Get the accepted and rejected proposals"""

        return self.ParticipantBehaviour.get_results()
