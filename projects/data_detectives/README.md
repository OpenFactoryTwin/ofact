# 🕵️‍♂️ Data Detectives 🕵️‍♀️- 🐛 Schwachstellen erkennen ⚠️

Die Aufträge in einem kleinen Produktionsbetrieb sind wieder einmal verspätet ⏰. Was sich auf dem Shop Floor beobachten lässt, stellt der Produktionsplaner auch im ERP-System fest 📊. Gleichzeitig beklagt der Disponent eine diffuse Überlastung einiger Maschinen 🏭 und des Personals 👷‍♂️. Zwar werden an verschiedenen Statuspunkten im Prozess Daten etwa durch Scans 📡 erhoben, jedoch können beide auf Anhieb in den Daten keine Ursache für das Problem erkennen. Es fehlt ihnen eine Möglichkeit, Zusammenhänge aus Aufträgen, Prozessen und Ressourcen abzubilden und so Transparenz bis hin zur Vorhersagbarkeit zu schaffen 🔮. Die Frustration steigt 😫, aber das muss sie ja gar nicht!

Um die Ursachen herauszufinden, ist euer Spürsinn gefragt 🕵️‍♀️🕵️‍♂️: Im moderierten Workshop legt ihr selbst Hand an unserem Analyse-Tool **Open Factory Twin (OFacT)** an 🛠️. Dabei zeigen wir euch zunächst, wie mithilfe von KI 🤖 aus einem Standarddatensatz aus Scan-Ereignissen ein Digitaler Zwilling entsteht 👫. Daten sammeln lohnt sich also! Mithilfe von Simulation 🖥️ werdet ihr Kennzahlen zu den Aufträgen, den Montageprozessen und den dafür benötigten und zur Verfügung stehenden Ressourcen analysieren 📈.

Für den Analysepfad von der aggregierten Auftragssicht bis hin zur einzelnen Ressource sind keine SQL- oder Programmierkenntnisse erforderlich 🚫💻. Wir sind gespannt, ähnliche Fragestellungen aus eurer Praxis und mögliche Lösungsansätze zu diskutieren 💡.

## Ablauf

- **1. Vorstellung**
  - Bicycle Factory
  - Unternehmensdaten
- **2. Kennenlernen**
  - Einführung
- **3. Ermitteln**
  - Schwachstellen erkennen

## Wie installiere ich OFacT über GitHub?

## Vorbereitungen 

Für den Workshop braucht ihr eine integrierte Entwicklungsumgebung (IDE), wie zum Beispiel **PyCharm** oder **VSCode**, sowie die empfohlene Python-Version (3.12) und eine aktuelle Git-Version. 
Wichtig ist auch, einen Python-Interpreter zu initialisieren.

Keine Sorge, mit einer beliebigen Suchmaschine ist dies einfach und schnell gemacht. 
Für diesen Workshop haben wir die notwendigen Installationen bereits vorbereitet, damit ihr möglichst viel Spaß mit unserem Analyse-Tool **OFacT** habt.

Öffnet bitte das vorinstallierte **PyCharm**. Dort findet ihr ein Projekt, das ihr nutzen könnt.

## Installation von OFacT

Um ein Gefühl dafür zu bekommen, wie einfach die Installation von **OFacT** ist, werden wir dies gemeinsam tun. Öffnet das Terminal (Alt + F12) und gebt folgenden Befehl ein:
- git clone https://github.com/OpenFactoryTwin/ofact.git .

**Voilà!** Ihr habt gerade **OFacT** installiert. Ihr könnt dies überprüfen, indem ihr die Ordnerstruktur betrachtet. Wenn ihr einen **OFacT**-Ordner findet, habt ihr alles erfolgreich abgeschlossen.

Environment erstellen und danach aktivieren ...

## Installation notwendiger Bibliotheken

Bibliotheken in Python sind Sammlungen von vorgefertigten Funktionen und Klassen, die wiederverwendbare Codebausteine bereitstellen, um spezifische Aufgaben zu erleichtern.

Öffnet erneut das Terminal (Alt + F12):

Die notwendigen Bibliotheken haben wir im Vorfeld definiert und stellen diese für den Open-Source-Code zur Verfügung. Der Terminalbefehl hierfür lautet:
- pip install -r requirements.txt

Sehr schön, Data Detectives! Ihr seid nun bereit, selbst Hand an unserem Analyse-Tool Open Factory Twin (OFacT) anzulegen 🕵️‍♀️🕵️‍♂️🛠️