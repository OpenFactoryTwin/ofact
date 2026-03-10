Twin
![Digital Twin Components](assets/imgs/digital_twin_components.png)

### Agent Control and State Model
The digital twin core consists of state model and the agent control. While the state model contains the state of the factory (could also include the change history represented as executed processes (list of ProcessExecution(s))), the agent control represents the control rules/ policies of the system. Meaning the state model and the agent control builds the center part of the OFacT. Additionally, the digital twin core contains the change handler, the model learning and repository services. While the change handler serves as layer between the agent control/ data integration and the state model, the model learning provides basic functionalities for machine learning and the repository services provides the capability to persist the state model.

The Digital Twin Core forms the central foundation of OFacT and consists of two primary components: - the State Model - and the Agent Control.

The State Model represents the complete state of the factory and can optionally include the change history represented as a list of executed processes (List<ProcessExecution>). The Agent Control, on the other hand, embodies the control rules and policies of the system, defining how the autonomous agents behave and make decisions within the factory environment. Together, the State Model and Agent Control build the center part of OFacT.

Additionally, the Digital Twin Core contains three essential supporting services: - the Change Handler, - Model Learning, - and Repository Services.

The Change Handler serves as an intermediary layer between the Agent Control and data integration on one side and the State Model on the other, ensuring all state changes are validated and consistent. Model Learning provides basic functionalities for machine learning, enabling the system to learn from historical data. Repository Services provide the capability to persist the State Model, ensuring long-term data retention, versioning, and recoverability of the entire system state.

<div align="center">
  <img src="twin.png" width="500" height="300" alt="Schematic Structure Twin" />
  <h3>Schematic Structure Twin</h3>
</div>
State Model
The State Model contains the data model of the environment (production and logistics). The idea is that each scenario has its own state model object. This means that no distinction is made between scenarios that reflect the physical world and virtually generated simulation scenarios. In both cases, changes from the environment are included in the state model in the same way (however, in some cases through passing the agents first). Next to environment the planning services such as the factory planner or the scenario analytics are based on the state model. While the "static" state model is the output of the factory planner, the state model serves as input for the scenario analytics. Concluding, the state model is connected to all areas of the digital twin framework.

Change Handler
The change_handler serves as the interface layer of the state model. Therefore, changes made by the agents (-control) or coming from the real world must pass the change handler.

Agent Control
The agent_control contains the tools to change the state of the state model. This is done by interaction of multiple agents (Multi Agent System).

Model Learning
The model_learning provides tools for the learning of process models in a standardized way. This supports the standardized and easy sample generation and the dataset creation.

Repository Services
The repository_service is used to persist the state model. In general, state models can be categorized into "static" and the "dynamic" ones (further details are available in the state_model folder). In the modelling phase, a static state model is created while, e.g., for simulation runs a dynamic state model evolves. For these different use cases, different persistence tools are provided.

Notice
This work is licensed under the CC-BY-4.0.