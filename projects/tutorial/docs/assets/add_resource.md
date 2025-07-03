# Duplicate/Add a new resource to the state model 

A common use case in modeling is adding and removing resources of a resource type that already exists.
E.g., the frame assembly is overloaded and a second station should be added to split the load.
However, to ensure that the resource is used in the simulation, the resource should also be added to the processes.
This is the topic of the current file.
The same steps used for adding a resource can also be applied to its removal.

1. Add a new resource.
    - Ensure that the resource attributes are defined correctly. 
      Including the modeling of new storages and process_execution_plans.
2. Edit the Transition Model
    - The resource should be a destination or an origin to be usable in the process flow
3. Edit the agent's model to ensure that the agents also use the resource.


# How to build up a new scenario environment for agents.

In general, we model the agents in the agents_model excel file.
ToDo: Should also be modeled from excel file.

The agent class names displayed in the agent model excel file, 
are mapped in "mapping classes" to existing python classes that represent the agents.
json file erstellen!

These agents manage a collection of agent behaviors (e.g., order management behavior).

