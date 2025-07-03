import asyncio

from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour
from ofact.twin.state_model.processes import ProcessExecution


class PartReservationBehaviour(DigitalTwinCyclicBehaviour):
    """
    Not in use anymore. Needed for one of the scenarios of the paper.
    """

    def __init__(self):
        """
        process_executions_to_forward: a list of process_executions_components that should be forwarded to another agent
        """
        super(PartReservationBehaviour, self).__init__()

        part_reservation_template = {"metadata": {"performative": "reserve",
                                                  "ontology": "PartReservation",
                                                  "language": "OWL-S"}}
        part_reservation_cancel_template = {"metadata": {"performative": "cancel",
                                                         "ontology": "PartReservation",
                                                         "language": "OWL-S"}}
        self.templates = [part_reservation_template, part_reservation_cancel_template]

        self.metadata_conditions = {"ontology": "PartReservation"}

    async def run(self):
        await super().run()

        msg_received = \
            await self.agent.receive_msg(self, timeout=10, metadata_conditions_lst=[self.metadata_conditions])

        if msg_received:

            msg_content, msg_sender, msg_ontology, msg_performative = msg_received
            asyncio.create_task(self._handle_part_reservation(msg_content, msg_performative))

    async def _handle_part_reservation(self, part_reservation, msg_performative):

        if msg_performative == "reserve":
            self.agent.reserve_parts_combined(part_reservation)

        elif msg_performative == "cancel_reservation":
            process_execution: ProcessExecution = part_reservation
            parts = process_execution.get_parts()
            process_execution_id = process_execution.identification
            self.agent.cancel_parts_reservation(parts, process_execution_id)
