"""
Event Log Standard defined by an enum class.
For most columns, it could be the case that the content required for the twin attribute
is distributed in more than one column in the input dataset.
The ID should be individual for each entity (part, resource).
The ID classification should be done manually.
"""

from enum import Enum
from abc import abstractmethod


class EventLogStandard(Enum):

    def __init__(self,
                 string: str,
                 description: str):
        self.string = string
        self.description = description

    @classmethod
    @abstractmethod
    def from_string(cls, string: str):
        pass


class EventLogStandardClasses(EventLogStandard):
    """
    Note: For references mostly the first entry is used.
    """

    ORDER = ("Order'",
             "Unique identifier for an order.")
    CUSTOMER = ("Customer'",
                "Unique identifier for an customer.")
    FEATURE = ("Feature'",
               "Unique identifier for a feature.")

    EXECUTION = ("ProcessExecution'",
                 "Unique identifier for an execution event.")
    PROCESS = ("Process'",
               "Unique identifier for a process.")

    EVENT_TYPE = ("EventType'",
                  "Type of the event.")

    ENTITY_TYPE = ("EntityType'",
                   "Type identifier for a entity.")

    STATIONARY_RESOURCE = ("StationaryResource'",
                           "Identifier for a resource. "
                           "There can be more than one resource participating in the process execution")
    RESOURCE_TYPE = ("EntityType'",
                     "Type identifier for a resource.")

    PART = ("Part'",
            "Identifier for a part. "
            "There can be more than one part used in the process execution")
    PART_TYPE = ("PartType'",
                 "Type identifier for a part.")

    @classmethod
    def from_string(cls, string: str):
        return EventLogStandardClasses[string]


class EventLogStandardAttributes(EventLogStandard):
    PROCESS = ("process",
               "Process of a execution.")

    ENTITY_TYPE = ("entity_type",
                   "Type of a entity.")

    # different times types are available
    EVENT_TIME_SINGLE = ("event_time_single",
                         "Timestamp for the event. Only one event time is expected for the execution.")
    EVENT_TIME_TRACE = ("event_time_trace",
                        "Timestamp for the event. Is used OR to EVENT_START_TIME and EVENT_END_TIME.")
    EXECUTION_START_TIME = ("executed_start_time",
                            "Start timestamp for the execution. Is used XOR to EVENT_TIME.")
    EXECUTION_END_TIME = ("executed_end_time",
                          "End timestamp for the execution. Is used XOR to EVENT_TIME.")

    ORIGIN = ("origin",
              "Unique identifier for the origin resource.")
    DESTINATION = ("destination",
                   "Unique identifier for the destination resource.")

    RESULTING_QUALITY = ("resulting_quality",
                         "Quality of the result.")

    INDIVIDUAL_ATTRIBUTE = ("Individual Attribute",
                            "individual attribute of an object")

    @classmethod
    def from_string(cls, string: str):
        return EventLogStandardAttributes[string]


class EventLogStandardHandling(EventLogStandard):
    NUMBER = ("Number of Referenced Objects",
              "Number of Referenced Objects associated")

    @classmethod
    def from_string(cls, string: str):
        return EventLogStandardHandling[string]


class EventLogOrderAttributes(EventLogStandard):
    ORDER_DATE = ("order_date",
                            "")
    RELEASE_DATE_PLANNED = ("release_date_planned",
                            "")
    RELEASE_DATE_ACTUAL = ("release_date_actual",
                           "")
    DELIVERY_DATE_PLANNED = ("delivery_date_planned",
                             "")
    DELIVERY_DATE_ACTUAL = ("delivery_date_actual",
                            "")

    PRODUCT_CLASSES = ("product_classes",  # ToDo: Can I replace them with a data driven approach?
                       "")

    @classmethod
    def from_string(cls, string: str):
        return EventLogOrderAttributes[string]


if __name__ == "__main__":
    print(EventLogStandardClasses.ORDER.value)
    print(EventLogStandardClasses.ORDER.string)
    print(EventLogStandardClasses.ORDER.description)
