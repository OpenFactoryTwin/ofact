<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<div align="center">

   <img src="docs/source/_static/ofact - logo.png" width="1395" height="679" />
   <h3>Open Factory Twin</h3> 
   <h4>Open source Digital Twin Framework for Production and Logistics</h4>

  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

<p align="center">
  <br />
  <a href="https://www.isst.fraunhofer.de/de/abteilungen/industrial-manufacturing/technologien/OFacT.html"><strong>Explore the docs »</strong></a>
  <br />
  <br />
  <a href="https://www.isst.fraunhofer.de/de/abteilungen/industrial-manufacturing/technologien/OFacT.html">View Demo</a>
  ·
  <a href="https://github.com/OpenFactoryTwin/ofact/issues/new?assignees=&labels=Bug%2CNeeds+Triage&projects=&template=bug_report.yml&title=BUG%3A+">Report Bug</a>
  ·
  <a href="https://github.com/OpenFactoryTwin/ofact/issues/new?assignees=&labels=Enhancement%2CNeeds+Triage&projects=&template=feature_request.yml&title=%F0%9F%9A%80+ENH%3A+">Request Feature</a>
</p>

***

## Table of Contents

- [About](#about)
- [Structure](#structure)
- [Getting Started](#getting-started)
    - [System Prerequisites](#system-prerequisites)
    - [Quick Start](#quick-start)
- [Release Notes](#release-notes)
- [Contributing](#contributing)
- [License](#license)
- [Developers](#developers)

***

## About

The *Open Factory Twin* (OFacT) project aims to provide a digital twin for production and logistics environments.
Digital Twins (DT) represent their environment as a virtual model of all relevant parts of the real system. OFact 
is meant to support the design, planning and operation control of discrete material flow systems and thus supporting 
management of the system during the whole life cycle.     

Coming from the **challenges** such as ...

* shorter production life cycles
* frequently changing demands
* complex supply chains
* increasing number of possible product variants
* regulations, legal requirements and restrictions

... companies are faced constantly with a complex decision that often needs a dynamic evaluation and comparison of various
scenarios. Often detailed simulation models are the only way to get a reliable evaluation of costs and performance. Data 
of the real world has to be integrated regularly into the simulation models to keep them up-to-date.  
In the design (or re-design) phase of a production system, the digital twin can be used to simulate different design 
alternatives and evaluate them even before the real system exists. When the real system is in operation, the digital twin
can be used in an iterative way between planning orders and resources and controlling the plan during operations dealing 
with disruptions in real time.

OFacT is based on a general state model that describes the state of the factory and the possible behaviors and can be 
used for all kinds of discrete material flow systems (e.g. assembly lines, flexible matrix production, job shops,
warehouses or even supply networks). 
The model consists of the following basic elements:
* **orders** (that describe the "customer" demand)
* **entities** (**resources** and **parts**) that describe the physical objects in the system with parts being
transformed based on processes that are executed by reusable resources 
* **processes** define the possible transformation of parts (and sometimes resources) in time, space physical attributes 
and quality

While the processes describe the possibility space of the material flow system, (planned and actual) **process executions** 
describe the concrete transformation in the past, present and future and can be seen as event logs that capture the 
dynamic behavior of the system. Process executions can be created by the real system - planned process execution are 
generated based on data from planning systems such as ERP or APS systems while actual process executions are generated 
based of sensor or event data. Planned process executions can also be created by the **multi-agent system** that controls 
the state model. The control logic of the digital twin is realised by agent behaviours and thus separated from the static
state model allowing for complex and flexible control behaviours. The separation of possibility space and actual behavior
as well as the separation of state mode and agent-based control allows for a flexible and modular design of the digital
twin that can be adapted to the specific requirements of the production system. Even more it facilitates the learning
of the model from the data of the real system. 

<div align="center">
  <img src="ofact/docs/assets/imgs/OFacT_Ecosystem.png" width="1000" height="600" />
  <h3>Digital Twin Ecosystem</h3> 
</div>

The OFacT framework consists of the following super components:
- **Environment**: provides components to interact with all kinds of environments (real and virtual) and to integrate data
  - **Data Integration**: provides tools to integrate data into the digital twin including consistency checks and update
  mechanisms
  - **Work Instruction**: provides tools to pass work instructions (planned process executions) back to the physical 
  world (closed loop system)
  - **Simulation**: is a virtual environment, that mimics the behavior of the real world and can be used to evaluate
    different scenarios or produce forecasts
  - **Data Space Connector**: provides the tools to connect the digital twin to the data space and to share the digital
    twin with other companies
- **Digital Twin**: provides the state model, the agent control
  - **State Model**: describes the state of the factory and the possible behaviors as well a passed and planned transformations
  - **Agent Control**: provides the control logic of the digital twin based on order and resource agents
- **Planning Services**: provides tools to generate the state model and to create and evaluate scenarios
  - **Scenario Generation**: provides the capabilities to create difference scenarios based on manual parameter variation, 
  optimization or even Artificial Intelligence such as reinforcement learning agents or generative models
  - **Scenario Analytics**: provides the tools to determine KPI's based on the state model, visualize and compare them

<p align="right">(<a href="#readme-top">back to top</a>)</p>

***

## Structure

This project uses a [monolithic repository approach](https://en.wikipedia.org/wiki/Monorepo) and
consists of different parts that are located in different subfolders of the `ofact` folder. 
Examples are use case-specific models and adaptions (currently only the twin models) 
are offered in the `projects` folder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

***

## Getting Started

Detailed getting started guides are described for every component in their dedicated `README`
file, located in the corresponding subfolders.

### System Prerequisites

The following things are necessary to run this application:

- tested on Python 3.10 and 3.12
- requirements.txt

### Quick Start

The first release of the open factory twin contains the data model (digital twin state model component) 
and therefore next to the agent control, one of the two core elements of the digital twin.
The state model can be filled with two sample use cases that can be found in the `projects` folder:
The models are provided in the `{project_name}/model/twin/` folder, modeled in Excel files.

### Quick Start with Docker-Compose 
---

**1. Build & Run Services**

```bash
docker-compose up --build
```

**2. Access Services**

* **Jupyter:** [http://localhost:8888](http://localhost:8888)
* **Voila:** [http://localhost:8866](http://localhost:8866)
* **PyJabber (XMPP):** Port `5222`

**3. Change Voila Notebook**
Edit the file path in `docker-compose.yml`:

```yaml
command: >
  sh -c "voila path/to/your_notebook.ipynb ..."
```

**4. Stop Services**

```bash
docker-compose down
```
---

#### Tutorial

The tutorial shows a small example shop floor of a board game factory 
that contains a subset of the elements existing in the state model.
In this example, three parts are taken from a warehouse and assembled by a worker in a packing station.

In this tutorial four topics are introduced:
1. Modelling
2. Data Integration
3. Analytics
4. Simulation

To start with the tutorial, click [here](projects/tutorial/A%20Warm%20Welcome%20to%20OFacT!.ipynb).

Some parts of the tutorial already contains the bicycle world, a more complex scenario ...

#### Bicycle World

A more advanced scenario in the context of Industrie 4.0 is offered with the bicycle world. 
Here, a modular and flexible assembly produces customized bicycles. 
The assembly stations can execute one or more processes (standardized),  
and the main product has a flexible assembly step sequences (routing flexibility), 
restricted only by the assembly priority chart of each product.
This projects contains two variants, one with and one without material supply.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Release Notes

As stated before, the first release, contains only the data model (state model).
However, soon further parts of the project will become open source.
The aim is to offer an example case (bicycle world) that can be simulated (agent control) 
and analyzed (scenario analytics).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Contributing

Contributions to this project are greatly appreciated! 
For more details, see the `CONTRIBUTING.md` file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## License

This work is licensed under the Apache 2.0 license. 
See `LICENSE` file for more information.

In the meantime, the project was created within the scope of
the [Center of Excellence Logistics and It](https://ce-logit.com/).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Developers

- Christian Schwede ([HSBI](https://www.hsbi.de/en) | [Fraunhofer ISST](https://www.isst.fraunhofer.de/en.html))
- Jan Cirullies ([FH Dortmund](https://www.fh-dortmund.de/index.php?loc=en) | [Fraunhofer ISST](https://www.isst.fraunhofer.de/en.html))
- Adrian Freiter ([Fraunhofer ISST](https://www.isst.fraunhofer.de/en.html))

- Roman Sliwinski ([HSBI](https://www.hsbi.de/en))
- Jannik Hartog ([Fraunhofer ISST](https://www.isst.fraunhofer.de/en.html))

- Niklas Müller ([Fraunhofer ISST](https://www.isst.fraunhofer.de/en.html))

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

If you have any further questions, please do not hesitate to contact us:

- christian.schwede@isst.fraunhofer.de
- adrian.freiter@isst.fraunhofer.de

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Notice
The documentation part of this work is 
licensed under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode) while the software part is 
licensed under Apache 2.0.