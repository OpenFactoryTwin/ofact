# Architecture Decision Records List


Please keep this up to date if you make any changes to an existing ADR or if you add a new one.
When you add new ADR's please use the given template which is based on [MADR](https://github.com/joelparkerhenderson/architecture-decision-record/blob/main/templates/decision-record-template-madr/index.md)

## List of ADR's

| Name of ADR                                           | Status of ADR     | Last Change       |
| ----------------------------------------------------- |:-----------------:| -----------------:|
| [Plattform of the MVP](/ADRs/Plattform of the MVP.md)   | accepted          | 2021.06.28        |
| [Regular DigitalTwin in the MVP](/ADRs/Regular_digitalTwin_for_MVP.md)   | accepted          | 2021.06.28        |
| [Object calls in MVP](/ADRs/Object_calls_in_MVP.md)   | accepted          | 2021.09.04        |

## TODO
**nachfolgende Designentscheidungen sollten noch in ADRs umgesetzt werden:**
* In der Lösungsversion Riotana V2 sollen Agenten, die nicht an eine Reale Ressource gekoppelt sind, möglicherweise als Microservices (zb. openshift) umgesetzt werden, um durch die geringere Komplexität flexibeler beim skalieren des Systems zu sein. Edge-Agenten hingegen sollen weiterhin Spade nutzen.
* Statt Komplexer Roboter-Ökonomie soll im MVP ein zentraler Steuerungsagent genutzt werden der den Aufbau und die Ressourcen kennt und Prozesse Organisiert
* 