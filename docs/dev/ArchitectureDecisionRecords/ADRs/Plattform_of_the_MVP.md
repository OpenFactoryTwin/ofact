# Plattform of the MVP

Deciders: Christian Schwede

Technical Story: Architekturbesprechung

## Context and Problem Statement
Die Komplexität, sowie Unsicherheiten bezüglicht der finalen Architektur erschweren die Entwicklung eines Durchstichs

## Decision Drivers <!-- optional -->

* Zeitnahe Umsetung eines Prototypen gefodert
* Lernen am Prototypen für das Finale Design gewünscht

## Considered Options

* Durchstich mit Node-red
* Durchstich mit Streampipes
* Durchstich mit Plain-Python und SPADE

## Decision Outcome

Durchstich mit Plain-Python und SPADE wird gewählt, da die anderen Ansätze sich als komplizierter herauskristalisiert 
haben

### Positive Consequences <!-- optional -->

* Eine schnelle Umsetzung wird begünstigt
* Weniger Technologie-Overhed im MYP

### Negative Consequences <!-- optional -->

* Skalierbarkeit ist nur sehr begrenzt möglich

## Pros and Cons of the Options <!-- optional -->

### Durchstich mit Node-red

* Gut, weil Flow-basiertes Design ist nahe an den Professionellen Tools (NIFI etc.)
* Gut, weil eine Umsetzung vieler Komponenten "Quick&Dirty" geschehen kann, später aber sukkzessive ersetzbar bleibt
* Schlecht, weil die Kommunikation über PyNodeRed mit dem Digitalen Zwilling nicht funktioniert hat
* Schlecht, weil die Skalierbarkeit in NodeRed begrenzt/aufwändig ist

### Durchstich mit Streampipes

* Gut, weil es eine Professionelle Lösung für fast alle unsere Anforderungen darstellt
* Schlecht, weil der Entwicklungsstand der Lösung noch nicht ausgereift genug war, sodass selbst simple Aufgabe nur mit 
  Mühe und Debugging zu realisieren waren


### Durchstich mit Plain-Python und SPADE

* Gut, weil die nötige Infrastrucktur bereits lauffähig ist
* Gut, weil spätere Funktionen voraussichtlich auch in Python implementiert werden, sodass eine Anpassung an die 
  Hauptversion einfach erfolgen kann
* Gut, weil die ausschließliche verwendung von Agenten mit SPADE eine Auseinandersetzung mit Microservices erspart  
* Schlecht, weil keine vorgefertigte Lösung genutzt wird = Wenig Support/Stackoverflow etc. 
  und Risiken das Rad neu zu erfinden
* Schlecht weil Microservices anstelle von manchen Agenten (KPI-Agenten etc.) einfacher skaliert werden könnten  

