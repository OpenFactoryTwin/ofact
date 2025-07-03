# Object calls in MVP

Deciders: Christian Schwede, Roman Sliwinski, Adrian Freiter

Technical Story: Freitagsmeeting

## Context and Problem Statement

Agenten müssen Daten untereinander austauschen. In V1 soll dazu ein shared digital twin genutzt werden (noch nicht näher definiert)
In der MVP version soll aber nicht entgegen der Zielvorstellung von Agenten vollzugriff auf den Digitalen Zwilling vergeben werden.
Wir suchen also den richtigen Ansatz für interne Logistik Probleme gesucht im Rahmen des MVP mit hinblick auf V1.

## Decision Drivers <!-- optional -->

* Prototyp soll so einfach wie möglich sein
* Bei der Implementierung soll so gehandelt werden, dass es später einfach weiterverwendet/ modifiziert werden kann


## Considered Options

* Objektreferenzen als String an andere Agenten übergeben - diese suchen sich dann das Objekt (Pointer-mäßig)
* Objekt serialisieren und Kopie übergeben
* Datenbank mit Zugriffsrechten für jeden Agenten
* Alle Agenten haben Vollzugriff auf den Digitalenzwilling mit allen Objekten


## Decision Outcome

Option 1 Wurde gewählt, aber musste angepasst werden:
Im Digitalen zwilling verwalten wir in einem Dict Objekte die von Agenten aufgerufen werden.
Als Referenz wird der Key zum Objekt per Spade-Message übergeben (Pointer in Python sind nicht zielführend!)
Der Agent kann eine Methode query_by_id() aufrufen um zugriff zu einem Objekt zu erreichen.

### Positive Consequences <!-- optional -->

* Umsetzung relativ einfach
* Der Weg über Nachrichten wird nicht umgangen - später kann der Nachrichteninhalt auch angepasst werden, wenn die Implementierung einen anderen Weg gehen sollte

### Negative Consequences <!-- optional -->

* Die Objekte im Digitalen zwilling sind so stark vernetzt, dass weitreichende Zugriffsrechte ausgegeben werden
* Zugriffe können nicht kontrolliert werden

## Pros and Cons of the Options <!-- optional -->

### [option 2]

* Gut, weil beliebige Skalierbarkeit ohne Lese-Schreib-Konflikte
* Gut, weil realitätsnah (Eigene Sichten der Agenten können voneinander abweichend sein)
* Schlecht, weil kompliziert auf aktuellem Stand zu halten
* Schlecht, weil viele Informationen vielfach gespeichtert werden

### [option 3]

* Gut, weil integrierte Zugriffskontollen etc. nutzbar werden
* Schlecht, weil Komplexität den MVP Umfang überschreitet (Zeitaufwand für Auswahl, Aufsetzten, Schnittstellen)

### [option 4]

* Gut, weil implementierung trivial ist
* Schlecht, weil unkontrollierter Vollzugriff viele Möglichkeiten schafft eine anschließende Adaptierbarkeit nicht möglich zu machen
* Schlecht, weil realitätsfern und kaum dezentralisierter.  
