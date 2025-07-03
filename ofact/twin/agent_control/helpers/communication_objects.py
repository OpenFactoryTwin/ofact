"""
The communication_object is used for the communication of greater objects.
To avoid the serializing - only the object identification_s is communicated as reference and the object is stored in
the agents_model
"""
from copy import copy, deepcopy
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    pass


class CommunicationObject:
    next_id = 0

    def get_next_id(self):
        next_id = deepcopy(CommunicationObject.next_id)
        CommunicationObject.next_id += 1

        return next_id

    def __init__(self, identification: int = None):
        """
        Base class for the communication_objects
        :param identification: unique id, if None is passed identification_s is generated automatically
        """
        if identification is None or identification < CommunicationObject.next_id:
            self.identification: int = self.get_next_id()
        else:
            self.identification: int = identification

    def get_msg_content(self):
        return self.identification


class ObjectCO(CommunicationObject):

    def __init__(self, content_object, identification=None):
        super(ObjectCO, self).__init__(identification=identification)
        self.content_object = content_object

    def get_msg_content(self):
        return self.content_object


class ListCO(CommunicationObject):

    def __init__(self, content_list: list, identification=None):
        super(ListCO, self).__init__(identification=identification)
        self.content_list: list = content_list

    def get_msg_content(self):
        return self.content_list


class DictCO(CommunicationObject):

    def __init__(self, content_dict: dict, identification=None):
        super(DictCO, self).__init__(identification=identification)
        self.content_dict: dict = content_dict

    def get_msg_content(self):
        return self.content_dict


class CoordinationCO(CommunicationObject):
    """Used to request the central coordination of some scheduling objects"""

    def __init__(self, process_executions_components: list, resources_preferences: dict, identification=None):
        super(CoordinationCO, self).__init__(identification=identification)
        self.process_executions_components: list = process_executions_components
        self.resources_preferences: dict = resources_preferences

    def get_msg_content(self):
        msg_content = (self.process_executions_components, self.resources_preferences)
        return msg_content


class AvailabilityInformationCO(CommunicationObject):

    def __init__(self, processes, start_time_stamp, round, identification=None):
        super(AvailabilityInformationCO, self).__init__(identification=identification)
        self.processes: List = processes
        self.start_time_stamp = start_time_stamp
        self.round = round

    def get_msg_content(self):
        msg_content = (self.processes, self.start_time_stamp, self.round)
        return msg_content


class TransportScheduleCO(CommunicationObject):

    def __init__(self,  start_time_begin_64, transport_process_executions, support_resource, identification=None):
        super(TransportScheduleCO, self).__init__(identification=identification)
        self.transport_process_executions: List = transport_process_executions
        self.support_resource = support_resource
        self.start_time_begin_64 = start_time_begin_64

    def get_msg_content(self):
        msg_content = (self.identification, self.transport_process_executions, self.support_resource, self.start_time_begin_64)
        return msg_content
