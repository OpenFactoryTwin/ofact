# Open Factory Twin (OFacT)

<div align="center">
  <img src="docs/assets/imgs/OFacT_Ecosystem.png" width="1000" height="600" />
  <h3>Digital Twin Ecosystem</h3> 
</div>

## Structure

As represented on the schematic representation above, the hearth of the framework is the **twin library**, 
located in the `twin` folder and containing the agent control and the state model.

**Subfolder overview:**

| Area                  | Component             | Brief Summary                                                                                                                           |
|-----------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| **env**               | *data_integration*    | provides the tools to integrate the data into the digital twin (data transformation)                                                    |
|                       | *work_instruction*    | provides the tools to pass work instructions back to the physical world (closed loop system)                                            |
|                       | *simulation*          | simulate the behavior of the real world based on the state model                                                                        |
|                       | *interfaces*          | provide interfaces for the data_integration as well as for the work_instructions |
| **twin**              | *state model*         | data model that describes the state of the factory as well as the possible behaviour                                                    |
|                       | *change_handler*      | used to keep track about the changes on the state model                                                                                 |
|                       | *agent control*       | enables the change of the digital twin by providing control rules that describe how the twin model                                      |
|                       | *model learning*      | provides possibilities to learn digital twin elements (e.g. process models, control rules)                                              |
|                       | *repository_service*  | to persist the digital twin (mainly the state model) can changed as well as change the system state                                     |
| **planning services** | *model_generation*    | used to create a static state model                                                                                                     |
|                       | *scenario_generation* | provides the capabilities to create difference scenarios based on parameter variation                                                   |
|                       | *scenario_analytics*  | provides the tools to determine KPI's based on the state model and visualize them                                                       |

***

## Notice
This work is licensed under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode).
