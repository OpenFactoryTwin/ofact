"""
Contains the objects that are used for the communication, respectively having the same understanding
Link to the FIPA contract net protocol explanation: http://www.fipa.org/specs/fipa00029/SC00029H.html
"""

from __future__ import annotations

from copy import copy
from enum import Enum
from typing import TYPE_CHECKING, Optional

from ofact.twin.agent_control.behaviours.planning.tree.process_executions_components import \
    ProcessExecutionsComponent, ProcessExecutionsVariant, ProcessExecutionsPath
from ofact.twin.agent_control.helpers.communication_objects import CommunicationObject
from ofact.twin.state_model.entities import Resource

# from dsplot.config import config
# from dsplot.graph import Graph

if TYPE_CHECKING:
    import spade
    from ofact.twin.agent_control.behaviours.planning.tree.preference import ProcessExecutionPreference
    from ofact.twin.state_model.entities import EntityType, Entity
    from ofact.twin.state_model.processes import ProcessExecution


class CallForProposal(CommunicationObject):  # Object

    class RequestType(Enum):
        FIX = "FIX"
        FLEXIBLE = "FLEXIBLE"

    def __init__(self, reference_cfp: CallForProposal | None, sender_name: str, client_object, order, preference,
                 request_type=RequestType.FIX, predecessor_cfp: CallForProposal | None = None,
                 issue_id=None, identification: int = None, node_path=None):
        """
        The call_for_proposal_object is used to share all relevant information in a negotiation and find an agreement.
        :param reference_cfp: a reference CallForProposal that is used to refer the CallForProposal (self)
        to another one (reference_cfp) that is the reason for the CallForProposal
        :param request_type: Type of the request if FIX, time_slots etc. are booked if FLEXIBLE only information
        are provided
        :param client_object: the client_object can be a WorkOrder or a ProcessExecution (the object that is responsible
        for the requesting)
        ensure that predecessor processes are performed and not necessary proposals are avoided
        :param request_object: the request_object determines the demand type and outcome of the negotiation
        if PROCESS_ORGANIZATION, the process and process demands are organized
        if RESOURCE, the resource is is transported to the location needed and blocked for the process
        if PART, the part availability is checked and the transport the demand location is organized
        :param preference: a function that describes the time_preferences of the order agent
        # ToDo :param amount_processes: the amount of (equal) processes needed
        #  for the moment to complex because the respective process_executions_components also needed
        #  (specification of parts_involved etc.)
        (for example more than one are needed for material_orders)
        :param issue_id: issues should be handled by the same resource/ if resource is a connector
        """
        super(CallForProposal, self).__init__(identification=identification)

        self.sender_name = sender_name
        self.reference_cfp: Optional[CallForProposal] = reference_cfp
        self.predecessor_cfp = predecessor_cfp  # ToDo: currently obsolete
        self.client_object = client_object
        self.order = order
        self.issue_id = issue_id  # should be individual for each issue (decision possible - )

        if node_path is None:
            # node_part_identification = ProcessExecutionsComponent.get_node_part_identification()
            # cfp_path = [(self.identification, node_part_identification)]
            node_path = [self.identification]
        else:
            node_path = copy(node_path)
            node_path.append(self.identification)
        self.node_path = node_path

        self.preference: ProcessExecutionPreference = preference
        self.request_type = request_type  # ToDo: obsolete?
        # self.amount_processes = amount_processes

        # ToDo: Use Case: Transport Reservation for the expected time needed to complete the order
        # ToDo: self.time_limit = time_limit


class ResourceCallForProposal(CallForProposal):

    def __init__(self, reference_cfp: CallForProposal | None, sender_name: str, client_object, order,
                 requested_entity_types: list[tuple[EntityType, int]], locations_of_demand: list[Resource],
                 preference: ProcessExecutionPreference, fixed_origin=None,
                 predecessor_cfp: CallForProposal | None = None,
                 long_time_reservation: None | dict[EntityType | Entity, float] = None, entity_types_storable=None,
                 request_type=CallForProposal.RequestType.FIX, issue_id=None, identification: int = None,
                 node_path=None):
        """
        :param requested_entity_types: the entity_types of a requested resources
        the requested resource (entity_type) needed
        :param entity_types_storable: at least one entity_type should be storable
        :param fixed_origin: to specify the origin resource (position) -> only the difference between origin resource
        and the location of demand should be planned
        :param locations_of_demand: location/ resource, where the resource is needed
        :param long_time_reservation: planned process_executions_components
        """
        super(ResourceCallForProposal, self).__init__(sender_name=sender_name, reference_cfp=reference_cfp,
                                                      predecessor_cfp=predecessor_cfp,
                                                      client_object=client_object, order=order, preference=preference,
                                                      request_type=request_type, issue_id=issue_id,
                                                      identification=identification, node_path=node_path)
        self.requested_entity_types: list[tuple[EntityType, int]] = requested_entity_types
        self.entity_types_storable: list[EntityType] = entity_types_storable
        self.locations_of_demand: list[Resource] = locations_of_demand
        self.fixed_origin: Resource = fixed_origin
        if long_time_reservation is None:
            long_time_reservation = {}
        self.long_time_reservation_duration: dict[EntityType, (int, bool)] = long_time_reservation

    def unpack(self):
        return self.reference_cfp, self.client_object, self.order, self.preference, \
            self.request_type, self.issue_id, self.requested_entity_types, self.entity_types_storable, \
            self.locations_of_demand, self.fixed_origin, self.long_time_reservation_duration, self.node_path


class ProcessCallForProposal(CallForProposal):

    def __init__(self, reference_cfp: CallForProposal | None, sender_name: str, client_object, order,
                 process_execution: ProcessExecution, preference: ProcessExecutionPreference,
                 long_time_reservation: dict[EntityType, (int, bool)], fixed_origin=None,
                 predecessor_cfp: CallForProposal | None = None,
                 request_type=CallForProposal.RequestType.FIX, issue_id=None, identification: int = None,
                 node_path=None):
        """
        :param process_execution: planned process_executions_components
        :param long_time_reservation: a long time reservation blocks the resource for a longer time
        (e.g. transport resource for the main_product)
        """
        super(ProcessCallForProposal, self).__init__(sender_name=sender_name, reference_cfp=reference_cfp,
                                                     predecessor_cfp=predecessor_cfp,
                                                     client_object=client_object, order=order, preference=preference,
                                                     request_type=request_type, issue_id=issue_id,
                                                     identification=identification, node_path=node_path)
        self._process_execution = process_execution
        self.fixed_origin: Resource = fixed_origin
        self.long_time_reservation: dict[EntityType, (int, bool)] = long_time_reservation

    @property
    def process_execution(self):
        return self._process_execution

    @process_execution.setter
    def process_execution(self, process_execution):
        if process_execution.process.name != self._process_execution.process.name:
            raise Exception(f"Wrong process execution: {process_execution.process.name} != "
                            f"{self._process_execution.process.name}")
        self._process_execution = process_execution

    def unpack(self):
        return (self.reference_cfp, self.client_object, self.order, self.preference, self.request_type, self.issue_id,
                self.process_execution, self.fixed_origin, self.long_time_reservation, self.node_path)

    def get_process_executions(self) -> list[ProcessExecution]:
        return [self.process_execution]


class ProcessGroupCallForProposal(CallForProposal):

    def __init__(self, reference_cfp: CallForProposal | None, sender_name: str, client_object, order,
                 process_executions: list[ProcessExecution], preference: ProcessExecutionPreference,
                 long_time_reservation: dict[EntityType, (int, bool)], fixed_origin=None,
                 predecessor_cfp: CallForProposal | None = None,
                 request_type=CallForProposal.RequestType.FIX, issue_id=None, identification: int = None,
                 node_path=None):
        """
        :param process_executions: planned process_executions_components
        :param long_time_reservation: a long time reservation blocks the resource for a longer time
        (e.g. transport resource for the main_product)
        """
        super(ProcessGroupCallForProposal, self).__init__(sender_name=sender_name, reference_cfp=reference_cfp,
                                                          predecessor_cfp=predecessor_cfp,
                                                          client_object=client_object, order=order,
                                                          preference=preference,
                                                          request_type=request_type, issue_id=issue_id,
                                                          identification=identification, node_path=node_path)
        self.process_executions = process_executions
        self.fixed_origin: Resource = fixed_origin
        self.long_time_reservation: dict[EntityType, (int, bool)] = long_time_reservation

    def unpack(self):
        return (self.reference_cfp, self.client_object, self.order, self.preference,
                self.request_type, self.issue_id, self.process_executions, self.fixed_origin,
                self.long_time_reservation, self.node_path)

    def get_process_executions(self) -> list[ProcessExecution]:
        return self.process_executions


class PartCallForProposal(CallForProposal):

    def __init__(self, reference_cfp: CallForProposal | None, sender_name: str, client_object, order,
                 requested_entity_types: list[tuple[EntityType, int]],
                 locations_of_demand: list[Resource], preference: ProcessExecutionPreference,
                 predecessor_cfp: CallForProposal | None = None,
                 request_type=CallForProposal.RequestType.FIX, issue_id=None, identification: int = None,
                 node_path=None):
        """
        :param requested_entity_types: the entity_types of a requested parts with respective demand_amount
        :param locations_of_demand: location/ resource, where the resource is needed
        """
        super(PartCallForProposal, self).__init__(sender_name=sender_name, reference_cfp=reference_cfp,
                                                  predecessor_cfp=predecessor_cfp,
                                                  client_object=client_object, order=order, preference=preference,
                                                  request_type=request_type, issue_id=issue_id,
                                                  identification=identification, node_path=node_path)
        self.requested_entity_types: [tuple[EntityType, int]] = requested_entity_types
        self.locations_of_demand = locations_of_demand

    def unpack(self):
        return self.reference_cfp, self.client_object, self.order, self.preference, \
            self.request_type, self.issue_id, self.requested_entity_types, self.locations_of_demand, self.node_path


class Proposal(CommunicationObject):
    class Status(Enum):
        OPEN = "OPEN"
        ACCEPTED = "ACCEPTED"
        REJECTED = "REJECTED"

    def __init__(self, call_for_proposal: CallForProposal, provider: str | spade.message,
                 status=Status.OPEN, identification: int = None):
        """
        A proposal is sent from the responder to the requester as action on a call for proposal
        :param call_for_proposal: the connected call_for_proposal object
        # ToDo: long-long-way: probabilities - assumption
        # ToDo: long-long-way: partial proposal
        """
        super(Proposal, self).__init__(identification)
        self.call_for_proposal: CallForProposal = call_for_proposal

        if type(provider) != str:
            provider = provider.localpart
        self.provider = provider

        self._status = status

        self.process_executions = {}

        self.preferred_time_slot = None
        self.price_preferred: None | float = None
        # ToDo: preferences

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        # if status == type(self).Status.ACCEPTED:
        #     if self.get_process_executions_components_lst():
        #         if (type(self).Status.ACCEPTED not in
        #                 [component.status for component in self.get_process_executions_components_lst()]):
        #             raise Exception("WHAT")
        self._status = status

    def accept(self):
        if self._status == type(self).Status.REJECTED:
            raise Exception
        self.status = type(self).Status.ACCEPTED

    def accepted(self):
        return bool(self.status == type(self).Status.ACCEPTED)

    def reject(self):
        self.status = type(self).Status.REJECTED

    # def get_time_slot(self, start_time, end_time):
    #     """After the proposal is accepted, the time_slots are determined
    #     This method returns the time_slot for the process_execution_component handled by the proposal"""
    #     if self.status == type(self).Status.OPEN:
    #         raise Exception
    #
    #     return self.process_executions_component.get_time_slot(start_time, end_time)

    def add_price_preferred(self, price_preferred):
        self.price_preferred = price_preferred

    # def _visualize(self):
    #     """Create and plot a visualization of the tree"""
    #
    #     graph_dict = {self.get_proposal_node_description(): [branch.get_proposal_node_description()
    #                                                          for branch in self.sub_proposals]}
    #     frontier_nodes = self.sub_proposals
    #     while True:
    #         new_frontier_nodes = []
    #         for branch in frontier_nodes:
    #             new_frontier_nodes.extend(branch.sub_proposals)
    #             graph_dict.setdefault(branch.get_proposal_node_description(),
    #                                   []).extend([branch.get_proposal_node_description()
    #                                               for branch in branch.sub_proposals])
    #
    #         if not new_frontier_nodes:
    #             break
    #         frontier_nodes = copy(new_frontier_nodes)
    #
    #     # graph = Graph(graph_dict, directed=False)
    #     # graph.plot(orientation=config.TREE_ORIENTATION,  # shape=config.LEAF_SHAPE,
    #     #            output_path='./debugging_images/component_tree.png')  # can be found in the project folder
    #
    # def get_proposal_node_description(self):
    #     return str(self.identification)


class ProcessExecutionsVariantProposal(Proposal, ProcessExecutionsVariant):

    def __init__(self, call_for_proposal: CallForProposal, provider: str | spade.message, status=Proposal.Status.OPEN,
                 identification: int = None,
                 issue_id=None, process_execution_id=None, goal=None, goal_item=None,
                 process_executions_components_parent=None, process_executions_components=None,
                 precondition_process_executions_components=None, reference_preference=None,
                 origin=None, destination=None,
                 node_identification=None, cfp_path=None, type_=ProcessExecutionsComponent.Type.STANDARD):

        Proposal.__init__(self, call_for_proposal=call_for_proposal, provider=provider, status=status,
                          identification=identification)
        ProcessExecutionsVariant.__init__(
            self, issue_id=issue_id, process_execution_id=process_execution_id, goal=goal,
            process_executions_components_parent=process_executions_components_parent,
            process_executions_components=process_executions_components,
            origin=origin, destination=destination,
            precondition_process_executions_components=precondition_process_executions_components,
            goal_item=goal_item, reference_preference=reference_preference,
            node_identification=node_identification, cfp_path=cfp_path, type_=type_)

    def overarching_proposal_rejected(self):
        """Check if all overreaching proposals are rejected"""

        if self.process_executions_components_parent is not None:
            rejected = bool(self.process_executions_components_parent.status == Proposal.Status.REJECTED)
        else:
            rejected = True
        return rejected

    def get_sub_proposals(self, type_=None):
        sub_proposals = self.get_process_executions_components_lst()

        if type_ is None:
            return sub_proposals

        if type_ == "OPENACCEPTED":
            sub_proposals_filtered = [sub_proposal for sub_proposal in sub_proposals
                                      if sub_proposal.status == type(self).Status.OPEN or
                                      sub_proposal.status == type(self).Status.ACCEPTED]
        elif type_ == "OPENREJECTED":
            sub_proposals_filtered = [sub_proposal for sub_proposal in sub_proposals
                                      if sub_proposal.status == type(self).Status.OPEN or
                                      sub_proposal.status == type(self).Status.REJECTED]
        elif type_ == "ACCEPTED":
            sub_proposals_filtered = [sub_proposal for sub_proposal in sub_proposals
                                      if sub_proposal.status == type(self).Status.ACCEPTED]
        elif type_ == "REJECTED":
            sub_proposals_filtered = [sub_proposal for sub_proposal in sub_proposals
                                      if sub_proposal.status == type(self).Status.REJECTED]
        else:
            raise NotImplementedError(type_)

        return sub_proposals_filtered

    def get_price_over_time(self):
        """Get the price for each time stamp and the duration to choose"""
        price = self.get_price()
        return price

    def update(self):
        """Update the process_executions_component because new information available"""
        print("STARt")
        self.update()
        self.get_price()

    def acceptable(self):
        acceptable = True
        if ((self.get_accepted_time_periods().size == 0 and not isinstance(self.call_for_proposal, PartCallForProposal)) \
                or self.status == type(self).Status.REJECTED):
            acceptable = False

        else:
            for goal, process_executions_components in self.process_executions_components.items():
                available = [pec for pec in process_executions_components if pec.status != type(self).Status.REJECTED]
                if not available:
                    acceptable = False
                    break

        return acceptable

    def add_preferred_time_slot(self, preferred_time_slot):
        """Add the preferred time slot """

        if preferred_time_slot[0] != "datetime64[s]" or preferred_time_slot[1] != "datetime64[s]":
            preferred_time_slot = (preferred_time_slot[0].astype("datetime64[s]"),
                                   preferred_time_slot[1].astype("datetime64[s]"))

        if not self.check_time_slot_accepted(preferred_time_slot):
            raise Exception(self.check_time_slot_accepted(preferred_time_slot))

        self.preferred_time_slot = preferred_time_slot

    def get_process_executions(self, type_, higher_level=True):
        """Get the process_executions that are needed to achieve the proposal"""
        # if not self.process_executions:
        if self.status == type(self).Status.ACCEPTED:
            time_slots = True
        else:
            time_slots = False

        process_executions = self._determine_process_executions(time_slots=time_slots, type_=type_,  # self.status.name,
                                                                higher_level=higher_level)

        # if type_ in self.process_executions:
        #     process_executions = self.process_executions[type_]
        # else:
        #     process_executions = []
        # process_executions = [pe
        #                       for pe in process_executions
        #                       if isinstance(pe, ProcessExecution)]
        return process_executions

    def _determine_process_executions(self, time_slots, type_, higher_level=False):
        """Get the process_executions defined in the process_executions_component
        :param time_slots: determine if the time_slots are set for the process_executions
        :param higher_level: means that the process_executions are requested from a higher level
        Therefore, they can also be open
        """

        # consider the sub proposals
        if not (self.status.name == type_ or higher_level and self.status == type(self).Status.OPEN):
            return []

        process_execution = self.get_process_execution(time_slots=time_slots)

        if higher_level:
            sub_proposals = self.get_sub_proposals(type_=f"OPEN{type_}")
        else:
            sub_proposals = self.get_sub_proposals(type_=type_)

        process_executions = [process_execution]
        for sub_proposal in sub_proposals:
            process_executions_batch = sub_proposal._determine_process_executions(time_slots, type_, higher_level)
            process_executions.extend(process_executions_batch)

        return process_executions

    def get_participating_process_executions(self, type_):
        if self.type.name != "CONNECTOR":
            return []

        process_executions = [self.goal_item]

        return process_executions

    def add_process_executions(self, process_executions, type_):
        if type(self).Status.REJECTED == type_ and not self.overarching_proposal_rejected():
            return
        self.process_executions.setdefault(type_,
                                           []).extend(process_executions)


class ProcessExecutionsPathProposal(Proposal, ProcessExecutionsPath):

    def __init__(self, call_for_proposal: CallForProposal, provider: str | spade.message, status=Proposal.Status.OPEN,
                 identification: int = None,
                 issue_id=None, process_execution_id=None, goal=None, goal_item=None,
                 process_executions_components_parent=None, process_executions_components=None,
                 reference_preference=None,
                 origin=None, destination=None, connector_objects=None,
                 path_link_type: ProcessExecutionsPath.LINK_TYPES = ProcessExecutionsPath.LINK_TYPES.FIXED,
                 node_identification=None, cfp_path=None, type_=ProcessExecutionsComponent.Type.STANDARD):
        Proposal.__init__(self, call_for_proposal=call_for_proposal, provider=provider, status=status,
                          identification=identification)
        ProcessExecutionsPath.__init__(
            self, issue_id=issue_id, process_execution_id=process_execution_id, goal=goal,
            process_executions_components_parent=process_executions_components_parent,
            process_executions_components=process_executions_components,
            origin=origin, destination=destination,
            goal_item=goal_item, reference_preference=reference_preference,
            node_identification=node_identification, cfp_path=cfp_path, type_=type_,
            connector_objects=connector_objects, path_link_type=path_link_type)

    def overarching_proposal_rejected(self):
        """Check if all overreaching proposals are rejected"""

        if self.process_executions_components_parent is not None:
            rejected = bool(self.process_executions_components_parent.status == Proposal.Status.REJECTED)
        else:
            rejected = True

        return rejected

    def get_sub_proposals(self, type_=None):
        sub_proposals = self.get_process_executions_components_lst()

        if type_ is None:
            return sub_proposals

        if type_ == "OPENACCEPTED":
            sub_proposals_filtered = [sub_proposal for sub_proposal in sub_proposals
                                      if sub_proposal.status == type(self).Status.OPEN or
                                      sub_proposal.status == type(self).Status.ACCEPTED]
        elif type_ == "OPENREJECTED":
            sub_proposals_filtered = [sub_proposal for sub_proposal in sub_proposals
                                      if sub_proposal.status == type(self).Status.OPEN or
                                      sub_proposal.status == type(self).Status.REJECTED]
        elif type_ == "ACCEPTED":
            sub_proposals_filtered = [sub_proposal for sub_proposal in sub_proposals
                                      if sub_proposal.status == type(self).Status.ACCEPTED]
        elif type_ == "REJECTED":
            sub_proposals_filtered = [sub_proposal for sub_proposal in sub_proposals
                                      if sub_proposal.status == type(self).Status.REJECTED]
        else:
            raise NotImplementedError

        return sub_proposals_filtered

    def get_price_over_time(self):
        """Get the price for each time stamp and the duration to choose"""
        price = self.get_price()
        return price

    def update(self):
        """Update the process_executions_component because new information available"""
        print("STARt")
        self.update()
        self.get_price()

    def acceptable(self):
        acceptable = True
        if ((self.get_accepted_time_periods().size == 0 and not isinstance(self.call_for_proposal, PartCallForProposal))
                or self.status == type(self).Status.REJECTED):
            acceptable = False
            return acceptable

        sub_proposals = self.get_process_executions_components_lst()
        if not sub_proposals:
            return acceptable

        sub_proposals_status = [component.status for component in sub_proposals]
        if type(self).Status.ACCEPTED not in sub_proposals_status and \
                type(self).Status.OPEN not in sub_proposals_status:
            acceptable = False

        return acceptable

    def add_preferred_time_slot(self, preferred_time_slot):
        """Add the preferred time slot """

        if preferred_time_slot[0] != "datetime64[s]" or preferred_time_slot[1] != "datetime64[s]":
            preferred_time_slot = (preferred_time_slot[0].astype("datetime64[s]"),
                                   preferred_time_slot[1].astype("datetime64[s]"))

        if not self.check_time_slot_accepted(preferred_time_slot):
            raise Exception(self.check_time_slot_accepted(preferred_time_slot))

        self.preferred_time_slot = preferred_time_slot

    def get_process_executions(self, type_, higher_level=True):
        """Get the process_executions that are needed to achieve the proposal"""
        # if not self.process_executions:
        if self.status == type(self).Status.ACCEPTED:
            time_slots = True
        else:
            time_slots = False
        process_executions = self._determine_process_executions(time_slots=time_slots, type_=type_,
                                                                higher_level=higher_level)  # self.status.name)

        # if type_ in self.process_executions:
        #     process_executions = self.process_executions[type_]
        # else:
        #     process_executions = []
        # process_executions = [pe
        #                       for pe in process_executions
        #                       if isinstance(pe, ProcessExecution)]
        return process_executions

    def _determine_process_executions(self, time_slots, type_, higher_level=False):
        """Get the process_executions defined in the process_executions_component
        :param time_slots: determine if the time_slots are set for the process_executions"""

        # consider the sub proposals
        if not (self.status.name == type_ or higher_level and self.status == type(self).Status.OPEN):
            return []

        if higher_level:
            sub_proposals = self.get_sub_proposals(type_=f"OPEN{type_}")
        else:
            sub_proposals = self.get_sub_proposals(type_=type_)

        process_executions = []
        for sub_proposal in sub_proposals:
            process_executions_batch = sub_proposal._determine_process_executions(time_slots, type_, higher_level)
            process_executions.extend(process_executions_batch)

        return process_executions

    def get_participating_process_executions(self, type_):
        if not self.status.name == type_:
            return []

        process_executions = []
        if self.process_executions_components_parent:
            process_execution = self.process_executions_components_parent.goal_item
            process_executions.append(process_execution)

        sub_proposals = self.get_sub_proposals(type_=type_)
        connector_process_executions = \
            [process_execution
             for sub_proposal in sub_proposals
             if sub_proposal.type.name == "CONNECTOR"
             for process_execution in sub_proposal.get_participating_process_executions(type_)]

        process_executions.extend(connector_process_executions)

        return process_executions

    def add_process_executions(self, process_executions, type_):
        if type(self).Status.REJECTED == type_ and not self.overarching_proposal_rejected():
            return
        self.process_executions.setdefault(type_,
                                           []).extend(process_executions)
