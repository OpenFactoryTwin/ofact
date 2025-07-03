"""
This file contains the supplier agents that are not implemented now ...
Supplier agents are responsible for filling the warehouse and used for a clear interface for the production system.
@last update: ?.?.2022
"""

# Imports Part 1: Standard Imports

# Imports Part 2: PIP Imports

# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour
from ofact.twin.agent_control.basic import DigitalTwinAgent


class SupplierResourceDigitalTwinAgent(DigitalTwinAgent):
    """
    # ToDo
    """

    def __init__(self, name: str, organization, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str):
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server,
                         ip_address_xmpp_server=ip_address_xmpp_server)

        # self.address_book = address_book
        # self.digital_twin = digital_twin
        # # PartTransformationNode: amount – Batch size (kleinst)
        # self.supplier_resource = supplier_resource
        # # Behavior: (Kanban, Lieferplan, by Order)

        # ToDo KANBAN: wird von entsprechendem Puffer Agenten angestoßen (Trigger)
        # Requirement: Delivery by order (required information: required_part - entity_type & batch size)

        # triggered by another agent - communication on the other direction also needed?

        # ToDo Delivery schedule: delivery according to a schedule
        # Requirement: Schedule (times, parts, lot sizes/ batch sizes)

        # communication to other agents not needed

        # ToDo General requirement
        # derivation of the required process and creation of an process_execution - good_receipt

        self.Supply = self.Supply()

    def copy(self):
        agent_copy = super(SupplierResourceDigitalTwinAgent, self).copy()

        agent_copy.Supply = self.Supply()

        return agent_copy

    # behaviours
    class Supply(DigitalTwinCyclicBehaviour):
        """
        # ToDo
        """

        async def run(self):
            await super().run()
            # ToDo

    async def setup(self):
        print(f"[{self.name}] Hello World! I'm agent {str(self.jid)} \n {type(self).__name__} \n")
        # templates
        # ToDo: What is necessary

        # add behaviors to the agent
        self.add_behaviour(self.Supply)
