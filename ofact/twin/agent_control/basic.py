"""
# ToDo: Communication inside the agent should not necessarily run through the xmpp-server - receive_msg, send_messages
This module is for the digital twin agent, who provides some functions that simplify and standardize,
among other the communication.
All agents inherit from the DigitalTwinAgent.
see https://spade-mas.readthedocs.io/en/latest/agents.html for agent library doc
@last update: 21.08.2023
"""
# Imports Part 1: Standard Imports
from __future__ import annotations

import json
import logging
from abc import ABCMeta, abstractmethod
from copy import copy, deepcopy
from typing import TYPE_CHECKING, List, Optional

# Imports Part 2: PIP Imports
try:
    import slixmpp
    from spade.agent import Agent
    from spade.message import Message
    from spade.template import Template
except ModuleNotFoundError:
    slixmpp = None
    class Agent:
        pass
    Message = None
    Template = None

# Imports Part 3: Project Imports
from ofact.twin.agent_control.helpers.debug_str import get_debug_str

if TYPE_CHECKING:
    from ofact.twin.agent_control.organization import Agents
    from ofact.twin.change_handler.change_handler import ChangeHandler

logger = logging.getLogger("basic-agents")


class DigitalTwinAgent(Agent, metaclass=ABCMeta):
    """
    The Agent provides some basic functions to lean the code of the Agents who inherits.
    Basically to create msg (message), tpl (template) and also to receive a msg (used for communication).
    """

    def __init__(self, name: str, organization: Agents, change_handler: ChangeHandler, password_xmpp_server: str,
                 ip_address_xmpp_server: str):
        """
        :param name: name of the agent (among others needed for the communication between the agents)
        :param organization: a model that contains all agents that exist in the environment and
        stores the negotiation_objects (essentially) needed for the communication between the agents
        :param change_handler: An object that is responsible for the communication between the agents and
        the external world. Especially for the exchange of process_executions_components and if necessary sensor data
        :param password_xmpp_server: needed for the access to the xmpp server
        :param ip_address_xmpp_server: the server address is needed for the communication with the server.
        """
        self._name = name
        self.ip_address_xmpp_server = ip_address_xmpp_server
        self.jid = self.create_jid()

        # connect to agent framework
        super().__init__(self.jid, password_xmpp_server)

        self.agents: Agents = organization
        self.change_handler: ChangeHandler = change_handler

        if not hasattr(self, "address_book"):
            self.address_book = {}

    def duplicate(self, external_name=False):
        agent_copy = self.copy()
        # ToDo: adapt the jid and the name - question: which name already exist

        return agent_copy

    def adapt_duplicate(self, agent_number):
        new_name = self.name.split("_agent")[0] + f"{agent_number}_agent"
        self.name = new_name
        self.jid = self.create_jid()
        super().__init__(self.jid, self.password)

    def add_instance_suffix(self, suffix):

        # update the name and the jid
        new_name = self.name + suffix
        self.name = new_name
        self.jid = self.create_jid()
        super().__init__(self.jid, self.password)

        if hasattr(self, "address_book"):
            # update the address book
            try:
                self.address_book = {object_: name + suffix
                                     for object_, name in self.address_book.items()}
            except Exception:
                raise Exception(f"Could not update the address book for {self.name}. "
                                f"Maybe the agent name is already used as label for a digital twin object.")

    @abstractmethod
    def copy(self):
        agent_copy = copy(self)

        return agent_copy

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def create_jid(self, ip_address_xmpp_server: str = None, name: str = None):
        """
        Creating a jid from the name and the XMPP_SERVER address.
        :param ip_address_xmpp_server: the server address is needed for the communication with the server.
        :param name: The name of an agent.
        :return a jid (address of an agent)
        """

        if name is None:
            name = self.name
        if ip_address_xmpp_server is None:
            ip_address_xmpp_server = self.ip_address_xmpp_server

        return name + "@" + ip_address_xmpp_server

    def create_message(self, message_input: dict, json_: bool = True):
        """
        The method is used to specify a message with  the content from the message_input.
            Messages are used to communicate with other Agents.
        :param message_input: a dict used to specify the message
        :param json_: convert the body into json-format if True
        :return specified spade message
        """
        if message_input["to"]:
            msg = Message()
            for md_key, md_value in message_input["metadata"].items():
                msg.set_metadata(md_key, md_value)
            msg = self.create_msg_or_tpl(msg, message_input, json_)
            return msg
        else:
            return None

    def create_template(self, template_input: dict, json_: bool = True):
        """
        The method is used to specify a template with  the content from the template_input.
            Templates are used to assign incoming messages to the right behaviour.
            Only messages which suit to a appropriate template are assigned.
        :param template_input: a dict used to specify the template
        :param json_: convert the body into json-format if True
        :return specified spade template
        """
        logging.debug(get_debug_str(self.name, "") + f" Create template!")

        template_input = deepcopy(template_input)  # to ensure that the agent can be instantiated a second time ...
        tpl = Template()
        tpl.metadata = {md_key: md_value
                        for md_key, md_value in template_input["metadata"].items()}
        tpl = self.create_msg_or_tpl(tpl, template_input, json_)

        return tpl

    def create_msg_or_tpl(self, msg_or_tpl, msg_or_tpl_input=None, json_: bool = True):
        """
        :param msg_or_tpl: spade message or template with metadata set in advance
        :param msg_or_tpl_input: the dict provide the components to specify the spade template or the message and
            can include the following elements: to: str, sender: str, body, thread: str, metadata: dict
        :param json_: convert the body into json-format if True
        :return a spade template or message
        """
        if msg_or_tpl_input is None:
            msg_or_tpl_input = {}

        # the metadata must be set in advance
        msg_or_tpl_input.pop("metadata")

        for attr_key, attr_value in msg_or_tpl_input.items():
            if attr_key == "to" or attr_key == "sender":  # monitor if the receiver/ sender address is complete

                if type(attr_value) == slixmpp.jid.JID:
                    attr_value = attr_value.local
                if type(attr_value) == str and "@" + self.ip_address_xmpp_server not in attr_value:
                    attr_value = self.create_jid(name=attr_value)
                if attr_key == "to":
                    msg_or_tpl.to = attr_value
                else:
                    msg_or_tpl.sender = attr_value
            elif attr_key == "body":  # convert the message to json-format
                if json_ is True:
                    attr_value = json.dumps(attr_value)
                msg_or_tpl.body = attr_value
            elif attr_key == "thread":
                msg_or_tpl.thread = attr_value
            else:
                print(get_debug_str(self.name, "") + f" "
                      f"The attribute named {attr_key} does not exist in a message or template")

        return msg_or_tpl

    async def receive_msg(self, behaviour, timeout=10, metadata_conditions: dict = None,
                          metadata_conditions_lst: Optional[List] = None, expected_sender=None):
        """
        The method is used to check if a received_msg is usable/ has reached the right destination and
        print some details about the message to facilitate the debugging.
        :return the received and checked message
        """
        msg_received = await behaviour.receive(timeout=timeout)
        # ToDo: comments used for debugging
        # if self.name == "gear_shift_brakes_as_agent":
        #     print("msg received: ", msg_received)
        #     if msg_received:
        #         print(msg_received.sender.localpart)
        #         if msg_received.sender.localpart == "coordinator_agent":
        #             print(metadata_conditions_lst)
        if not metadata_conditions_lst:
            metadata_conditions_lst = [metadata_conditions]

        for metadata_conditions in metadata_conditions_lst:
            if msg_received:  # add conditions
                for condition, value in metadata_conditions.items():
                    if msg_received.metadata[condition] != value:
                        continue
                    if expected_sender:
                        sender = msg_received.sender
                        if type(sender) == slixmpp.jid.JID:
                            sender = sender.local
                        if sender != expected_sender:
                            continue

                # print(get_debug_str(self.name, behaviour.__class__.__name__) +
                #       f" has received a {msg_received.metadata['performative']} "
                #       f"message with content '{msg_received.body}' from {msg_received.sender}!")
                # debug json.loads(msg_received.body)

                msg_body_content = json.loads(msg_received.body)
                msg_content = behaviour.agent.agents.get_communication_object_by_id(msg_body_content)
                if msg_content is not None:
                    msg_content = msg_content.get_msg_content()
                msg_sender = msg_received.sender.local
                msg_ontology = msg_received.metadata["ontology"]
                msg_performative = msg_received.metadata["performative"]

                msg_with_metadata = (msg_content, msg_sender, msg_ontology, msg_performative)
                return msg_with_metadata
                # else:
                #     pass
                    # print(f"{self.jid} has not received any message after {timeout} seconds")
            # else:
            #     pass
                # print(f"{self.jid} has not received any message after {timeout} seconds")

        if msg_received is not None:
            return await self.receive_msg(behaviour, timeout, metadata_conditions, metadata_conditions_lst,
                                          expected_sender)

    async def send_msg(self, behaviour, receiver_list: list[str], msg_body: str, message_metadata: dict,
                       json_: bool = True):
        """
        Used to send (different) messages to a distribution list.
        :param behaviour: the behaviour which should send the message
        :param receiver_list: list of the names of the receiver
        :param msg_body: list of msg_contents
        :param message_metadata: metadata of the message
        :param json_: convert the body into json-format if True
        """

        if hasattr(msg_body, "identification"):
            self.agents.store_communication_object(msg_body)  # preferred way
            msg_body = msg_body.identification

        msg_body_metadata = {"body": msg_body, "metadata": message_metadata}
        for receiver in receiver_list:
            msg_d = {"to": receiver, **msg_body_metadata}
            msg_resource_sent = self.create_message(msg_d, json_)
            # if receiver == self.name:
            #     # use the direct way
            #     # self._pass_msg(msg_resource_sent)
            #     await behaviour.send(msg_resource_sent)
            #     continue
            await behaviour.send(msg_resource_sent)

    def _pass_msg(self, content):
        # find the right behaviour
        # idea: develop  a switch case for faster msg passing
        # self.submit(behaviour.enqueue(msg))  # find the right behaviour
        pass

    async def send_messages(self, behaviour, receiver_list: list[str], msg_body_list: list[str],
                            message_other_data: dict, json_: bool = True):
        """
        Used to send (different) messages to a distribution list.
        :param behaviour: the behaviour which should send the message
        :param receiver_list: list of the names of the receiver
        :param msg_body_list: list of msg_contents
        :param message_other_data: metadata of the message
        :param json_: convert the body into json-format if True
        """
        receivers_dict = {"to": receiver for receiver in receiver_list}
        msg_bodies_dict = {"body": msg_body for msg_body in msg_body_list}
        for receiver_dict in receivers_dict:
            for msg_body_dict in msg_bodies_dict:
                msg_resource_sent = self.create_message({**receiver_dict, **msg_body_dict, **message_other_data}, json_)
                await behaviour.send(msg_resource_sent)

    async def setup(self):
        print(get_debug_str(self.name, "") + f" Hello World! I'm agent {str(self.jid)} \n {type(self).__name__} ")

# ToDo: integrate change handler capabilities
