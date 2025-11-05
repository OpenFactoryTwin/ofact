"""Contains the resource binder ..."""

# Imports Part 3: Project Imports
from ofact.twin.agent_control.behaviours.basic import DigitalTwinCyclicBehaviour


class ResourceBindingBehaviour(DigitalTwinCyclicBehaviour):
    """Bind a resource to an order to ensure that the resource is not planned for any other order/ process."""

    def __init__(self):
        """
        process_executions_to_forward: a list of process_executions_components that should be forwarded to another agent
        """
        super(ResourceBindingBehaviour, self).__init__()

        release_template = {"metadata": {"performative": "inform",
                                         "ontology": "ResourceBinding",
                                         "language": "OWL-S"}}
        self.templates = [release_template]

        self.metadata_conditions = {"performative": "inform", "ontology": "ResourceBinding"}

    async def run(self):
        await super().run()

        msg_received = \
            await self.agent.receive_msg(self, timeout=10, metadata_conditions=self.metadata_conditions)

        if msg_received:
            msg_content, msg_sender, msg_ontology, msg_performative = msg_received
            self._handle_resource_binding(msg_content)

    def _handle_resource_binding(self, resource_binding):
        resource, binding_order = resource_binding
        if binding_order:
            self.agent.bind_resource(resource, binding_order)
            # print(f'\033[32mbinding: {resource.name} {binding_order.identification}\033[0m')
        elif binding_order is None:
            self.agent.unbind_resource(resource)
            # print(f'\033[34munbinding: {resource.name}\033[0m')

