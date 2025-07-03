# Specifying the Data Integration Pipeline

In the context of the digital twin, the data integration is used to integrate data from the data sources
of the physical world to the digital twin. </br>
Data sources can be IT tools (data on a higher level) and sensor data (low level):

- **Sales IT tool** level (ERP systems such as SAP or Odoo)
- **Shop Floor IT Tool** level (MES or PPC tools such as ...)
- **Sensor** level (such as temperature or position)

The available data is checked according relevance for the digital twin.
The relevant data should be subsequently integrated in the digital twin.
Therefore, the data sources and their content are mapped in an Excel file.

The Excel file should have the following sheets filled:

## adapter allocation

> The adapter allocation is used to define the data sources and the access point to these sources (path and Input (
> -type)). Additionally the priority of the data sources is defined to ensure a fast data integration.

| path                                                        | name space | priority | Input | Mapping    |
|-------------------------------------------------------------|------------|----------|-------|------------|
| 192.168.0.45/server/riotana_data/auftraege.xlsx             | Schmaus    | 10       | Excel |            |
| projects/Schmaus/models/twin/data_sources_v2/positionen.csv | Schmaus1   | 20       | csv   | positionen |
| RIOTANA_Fließband                                           | Schmaus    | 30       | MSSQL |            |

path: should contain a path that is reachable for the riotana-docker service. Input contains the format RIOTANA should
expect to find at the destination. currently supported Formats are: Excle-sheets (xlsx, xls)

## time restriction columns

> In the time restriction is used to restrict the input data according to read only a time window of the complete input
> data.

| mapping   | columns          | None values accepted | Min one column filled |
|-----------|------------------|----------------------|-----------------------|
| auftraege | Erstellzeitpunkt | FALSE                | TRUE                  |

## column mappings

> The column mapping is used to map the entries of the data sources to digital twin objects and their attributes.

| mapping              | external                         | mapping identification                     | mapping reference                                                                                          | class                                               | attribute                                                                 | handling                                                                                                                                | depends on                                                                             |
|----------------------|----------------------------------|--------------------------------------------|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| positionen           | AID                              | 18                                         | 20                                                                                                         | Order                                               | order                                                                     |                                                                                                                                         |                                                                                        |
| positionen           | AID                              | 19                                         | 20                                                                                                         | Part                                                | parts_involved                                                            | domain specific                                                                                                                         |                                                                                        |
| positionen           | PID                              | 20                                         |                                                                                                            | ProcessExecution                                    | identification                                                            |                                                                                                                                         |                                                                                        |
| positionen           | ArtNr                            | 21                                         | 29                                                                                                         |                                                     | ArtNr                                                                     | domain specific attribute                                                                                                               |                                                                                        |
| positionen           | Soll-Menge                       | 22                                         | 29                                                                                                         |                                                     | Soll-Menge                                                                | domain specific attribute                                                                                                               |                                                                                        |
| positionen           | Ist-Menge                        | 23                                         | 29                                                                                                         |                                                     | Ist-Menge                                                                 | domain specific attribute                                                                                                               |                                                                                        |
| positionen           | Soll-Lagerplatz                  | 24                                         | 29                                                                                                         |                                                     | Soll-Lagerplatz                                                           | domain specific attribute                                                                                                               |                                                                                        |
| positionen           | Ist-Lagerplatz                   | 25                                         | 29                                                                                                         |                                                     | Ist-Lagerplatz                                                            | domain specific attribute                                                                                                               |                                                                                        |
| positionen           | Bahnhof                          | 26                                         |                                                                                                            | WorkStation                                         | identification                                                            |                                                                                                                                         |                                                                                        |
| positionen           | Bahnhof                          | 27                                         | 17                                                                                                         | Feature                                             | features_requested                                                        | domain specific                                                                                                                         |                                                                                        |
| positionen           | Bahnhof                          | 28                                         | 20                                                                                                         | Process                                             | process                                                                   | domain specific                                                                                                                         |                                                                                        |
| positionen           | Bahnhof                          | 29                                         | 20                                                                                                         | Part                                                | parts_involved                                                            | domain specific                                                                                                                         | 32                                                                                     |
| positionen           | Bahnhof                          | 30                                         | 20                                                                                                         | WorkStation                                     | main_resource                                                             |                                                                                                                                         |                                                                                        |
| positionen           | Bahnhof                          | 31                                         | 20                                                                                                         | WorkStation                                     | resources_used                                                            |                                                                                                                                         |                                                                                        |
| positionen           | Bahnhof                          | 32                                         | 20                                                                                                         | WorkStation                                     | origin                                                                    | domain specific                                                                                                                         | 17                                                                                     |
| positionen           | Bahnhof                          | 32                                         | 20                                                                                                         | WorkStation                                     | destination                                                               | domain specific                                                                                                                         | 17                                                                                     |
| -------------------- | -------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| source of the data   | name of the external attribute   | unique identification for each attribute   | referene to the "mapping identification" - meaning that the attribute is mapped to the referenced object   | Digital Twin class, the attribute is an object of   | the attribute name (if identification, an object of "class" is created)   | handling can be used to apply special handling of the entry (possibilities are "domain specific", "domain specific attribute", ...)     | used for individual handling of the object depending on an other information (entry)   |

## split

The split operator can be used for entries that have more than one attribute in one entry element

| mapping            | external                       | separator           | action                                    | operation id             |
|--------------------|--------------------------------|---------------------|-------------------------------------------|--------------------------|
| positionen         | Bahnhof                        | ,                   | add_row                                   | 1                        |
| positionen         | Soll-Lagerplatz                | *                   | choose                                    | 1                        |
|                    |                                |                     |                                           |                          |
| source of the data | name of the external attribute | separation operator | Defines what todo with the different rows | ToDo: What does it mean? |

## clean up

> Used for imputation or dropping of the entry

| mapping    | external         | old value | replacing value | delete |
|------------|------------------|-----------|-----------------|--------|
| auftraege  | BasisRoutingCode | NULL      | Unknown         | False  |
| positionen | Bahnhof          | NULL      | Sonstige        | False  |

## static refinements

> Used if some objects every time have the same attribute values that are not delivered by the data sources but are
> needed, they can be imputed through static refinements.

| mapping            | type                                                            | class                                                | attribute                   | value                               |
|--------------------|-----------------------------------------------------------------|------------------------------------------------------|-----------------------------|-------------------------------------|
| positionen         | ProcessExecution.EventTypes                                     | ProcessExecution                                     | event_type                  | PLAN                                |
| positionen         |                                                                 | ProcessExecution                                     | resulting_quality           | 1                                   |
| positionen         |                                                                 | ProcessExecution                                     | source_application          | Schmaus                             |
| positionen         |                                                                 | ProcessExecution                                     | connected_process_execution | None                                |
| fließband          | ProcessExecution.EventTypes                                     | ProcessExecution                                     | event_type                  | PLAN                                |
| fließband          |                                                                 | ProcessExecution                                     | resulting_quality           | 1                                   |
| fließband          |                                                                 | ProcessExecution                                     | parts_involved              | None                                |
| fließband          |                                                                 | ProcessExecution                                     | source_application          | Schmaus                             |
| fließband          |                                                                 | ProcessExecution                                     | connected_process_execution | None                                |
|                    |                                                                 |                                                      |                             |                                     |
| source of the data | needed if the attribute itself is an object (from class "type") | class of the object from the attribute to be imputed | attribute to be imputed     | imputation value                    |
|                    |                                                                 |                                                      |                             |                                     |
| source of the data | name of the external attribute                                  | value to impute                                      | imputation value            | decide if the value is to delete .. |

## filters

> Used to include or exclude entries based on their values, to avoid that unexpected data leads to problems …

| mapping            | external                       | needed entries                                                                                                                                                 | not needed entries                                                                 | contains                                                                  |
|--------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| auftraege          | Durchfuehrungsart              | Behälter                                                                                                                                                       |                                                                                    |                                                                           |
| auftraege          | Durchfuehrungsart              | ""                                                                                                                                                             |                                                                                    |                                                                           |
| positionen         | Bahnhof                        |                                                                                                                                                                | A                                                                                  |                                                                           |
| positionen         | Bahnhof                        |                                                                                                                                                                |                                                                                    | B                                                                         |
| positionen         | Bahnhof                        |                                                                                                                                                                |                                                                                    | F                                                                         |
| positionen         | Bahnhof                        |                                                                                                                                                                |                                                                                    | H                                                                         |
| positionen         | Bahnhof                        |                                                                                                                                                                |                                                                                    | I                                                                         |
| positionen         | Bahnhof                        |                                                                                                                                                                |                                                                                    | K                                                                         |
| positionen         | Bahnhof                        |                                                                                                                                                                |                                                                                    | L                                                                         |
| positionen         | Bahnhof                        |                                                                                                                                                                |                                                                                    | N                                                                         |
| positionen         | Bahnhof                        |                                                                                                                                                                |                                                                                    | V                                                                         |
| positionen         | Bahnhof                        |                                                                                                                                                                |                                                                                    | Z                                                                         |
| positionen         | PID                            |                                                                                                                                                                |                                                                                    | EL                                                                        |
| positionen         | PID                            |                                                                                                                                                                |                                                                                    | LA                                                                        |
| positionen         | AID                            | ods*(auftraege; (1.0, "np.nan", "Order", "identification", "np.nan", "np.nan"))                                                                                |                                                                                    |                                                                           |
|                    |                                |                                                                                                                                                                |                                                                                    |                                                                           |
| source of the data | name of the external attribute | if filled, the entry is only considered, if the attribute value is equal to the "needed entries"                                                               | if filled, the entries that have the value "not needed entries" are not considered | if filled, the entries that have only the value "contains" are considered |
|                    |                                | ods*(auftraege; (1.0, "np.nan", "Order", "identification", "np.nan", "np.nan")): here the only orders that are also in another source (mapping) are considered |                                                                                    |                                                                           |

## sort by

> Used to sort the input data and ensure a chronological and therefore fast handling of the data integration.

| mapping            | external                                                        |
|--------------------|-----------------------------------------------------------------|
| auftraege          | Erstellzeitpunkt                                                |
| positionen         | Prozessstartzeitpunkt                                           |
| fließband          | Timestamp                                                       |
|                    |                                                                 |
| source of the data | name of the external attribute that serves as basis for sorting |

## aggregation

> Define in which combinations the data sources are combined and therefore the decentral information can be combined.

| first              | second             |
|--------------------|--------------------|
| positionen         | fließband          |
| auftraege          | positionen         |
| auftraege          | fließband          |
|                    |                    |
| source of the data | source of the data |
