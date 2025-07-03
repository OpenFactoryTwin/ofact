
# JSON Format Riotana

In diesem Dokument wird die Entwicklung des JSON Formats für die Kommunikation zwischen Front- und Backend in RIOTANAV2 dokumentiert.

## ABLAUF INTERAKTION


 1. Das Frontend fragt unter GET /filters nach allen verfügbaren Filtern im gewählten Zeitraum nach.
 2. Das Frontend holt sich der Reihe nach die ersten Daten von der API.
	- z.B. Orders
3. Wählt Nutzer Filter aus und klickt auf aktualisieren, so wird der zuständige Endpunkt samt Filterobjekt übergeben.
	- Das Auswählen der Filter geschieht dabei komplett im Frontend. 
	- "Nicht anzeigen" von gewissen Filtern aufgrund von übergeordneten Filtern (z.B. Orders) wird ebenso im Frontend realisiert.

4. Wird der Betrachtungszeitraum geändert, so müssen die Filter aktualisiert werden und die aktuelle Kategorie an Daten neu angefragt werden (orders, processes...)

## JSON - SCHEMA

dateStart und dateEnd sind UNIX-Timestamps und sind bei jeder Anfrage von Datensätzen erforderlich.

Schnellübersicht:

Endpunkt | Methode | Links
--- | --- | ---
/filter | GET | [Link](#filter)
/orders | GET | [Link](#orders)
/products | GET | [Link](#products)
/processes | GET | [Link](#processes)
/resources | GET |[Link](#resources) 
---

### Filter
**GET /filters?{dateStart}&{dateEnd}**
Returns:
```
{
	"orders": {
		"id": "string"
		"referenceValue: "string"
	},
	"products": {
		"id": "string"
		"referenceValue: "string"
	}
	"processes": {
		"id": "string"
		"referenceValue: "string"
	}
	"resources": {
		"id": "string"
		"referenceValue: "string"
	}
}
```
### Orders
**GET /orders?filter={...}**

{...} =
```
{
	dateStart: "number",
	dateEnd: "number",
	orders: [ "string" ] //id's der aufträge, welche angezeigt werden sollen
}
```
Returns:
```
{
	"orders": 
	[
		{
			"id": "string",
			"referenceValue": "string",
			"numberOfPiecesAbsolute": "number",
			"numberOfPiecesRelative": "number",
			"customer": "string",
			"startingDate": "number",
			"completionDate": "number",
			"orderStatus": "string",
			"priority": "number",
			"deliveryReliability": "string",
			"totalLeadTime": "number",
			"totalWaitingTime": "number",
			"currentStock": "number",
			"quality": "number",
			"performance": "number"
		}
	]
}
```

### Products
**GET /products?filter={...}**

{...} =
```
{
	dateStart: "number",
	dateEnd: "number",
	products: [ "string" ] //id's der produkte, welche angezeigt werden sollen
}
```
Returns:
```
{
	"products": 
	[
		{
			"id": "string",
			"referenceValue": "string",
			"targetQuantity": "number",
			"quantityProduced": "number",
			"differencePercentage": "number",
			"productShares": "number",
			"deliveryReliability": "string",
			"leadTime": "number",
			"totalLeadTime": "number",
			"waitingTime": "number",
			"totalWaitingTime": "number",
			"currentStock": "number",
			"quality": "number",
			"performance": "number"
		}
	]
}
```

### Prozesse
**GET /processes?filter={...}**

{...} = 
```
{
	dateStart: "number",
	dateEnd: "number",
	processes: [ "string" ] //id's der prozesse, welche angezeigt werden sollen
}
```
Returns:
```
{
	"processes": 
	[
		{
			"id": "string",
			"referenceValue": "string",
			"absoluteFrequency": "number",
			"processShare": "number",
			"deliveryReliability": "string",
			"leadTime": "number",
			"totalLeadTime": "number",
			"waitingTime": "number",
			"totalWaitingTime": "number",
			"quality": "number",
			"performance": "number"
		}
	]
}
```

### Ressourcen
**GET /resources?filter={...}**

{...} = 
```
{
	dateStart: "number",
	dateEnd: "number",
	resources: [ "string" ] //id's der ressourcen, welche angezeigt werden sollen
}
```
Returns:
```
{
	"resources": 
	[
		{
			"id": "string",
			"referenceValue": "string",
			"processFrequency": "number",
			"resourceShare": "number",
			"deliveryReliability": "string",
			"leadTime": "number",
			"waitingTime": "number",
			"stock": "number",
			"quality": "number",
			"performance": "number",
			"totalResourceUtilisation": "number",
			"kpiOre": "number"
		}
	]
}
```
