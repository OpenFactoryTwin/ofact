Exporter
========

The overall idea is to write an exporter that takes in a mapping file
that constitutes the target schema type and the structure of the schema
originating in the source object (i.e. the ``DigitalTwin`` object) which
implements ``dict_serialize``. If the target schema is *xlsx* the
mapping file should provide a *sheets* attribute that is a list of all
sheets with its corresponding columns and mappings to the source object.
A column mapping would consist of the column name as the key and the
keys needed to access the value in the source object. When a list is
provided the source object is accessed incrementally. I.e.
``["entity_type", "name"]`` would lead to
``source["entity_type"]["name"]``. The accessed value is stringified
subsequently and written to the target table.

Defining sheets
---------------

By defining the ``target_schema`` as *xlsx* a *sheets* attribute can be
provided as a list of sheets that should be included in the output of
the exporter. A sheet definition follows the following scheme:

::

   serialization_kind: "dict_flatten" | "list" | "single_value" | "mixed"
   serialize_unique: bool
   name: string
   source: string | list string
   filter: string
   source_function: string
   columns: list column
   start_row: int

Serialization kind
~~~~~~~~~~~~~~~~~~

To address different types of source attributes for serialization a
serialization kind is introduced:

-  ``"serialization_kind" : "dict_flatten"`` -> Accepts source
   attributes of type ``dict[any, list[Serializable]]`` and flattens the
   ``dict`` whose keys map to lists to a single list. The list is
   serialized into the sheet
-  ``"serialization_kind" : "list"`` -> Accepts source attributes of
   type ``list[Serializable]`` and serializes them directly to a table.
-  ``"serialization_kind": "single_value"`` -> Accepts a source
   attribute that is ``impl Serializable`` and writes it to the target
   file. This is equivalent to ``"list"`` where the source list has one
   entry.
-  ``"serialization_kind": "mixed"`` -> Accepts a source attribute that
   either meets the criteria of ``dict_flatten`` or ``list`` and reduces
   them to a single list which is being written to the target file.

Serialize unique
~~~~~~~~~~~~~~~~

When ``serialize_unique`` is set to ``True`` the table only consists of
unique values with an additional column ``amount`` that represents the
amount of identical entries in the source.

Source
~~~~~~

When a string is provided the provided input object is indexed by the
string and is subsequently used as the source for the sheet. Providing a
list of strings multiple source attributes of the input object can be
used for the same sheet.

Filter
~~~~~~

For each sheet a ``filter`` function can be provided to filter the
source attribute. The filter function has to be supported by the
exporter. The function has to return either ``True`` or ``False``.

Source function
~~~~~~~~~~~~~~~

Source attributes can be mapped using a source function. The source
function has to be supported by the exporter.

Start row
~~~~~~~~~

Specifies the first row which is used for writing to the target table.

Columns
~~~~~~~

A list of columns for the respective sheet. A column definition follows
this schema:

::

   column_kind: "simple" | "complex" | "fixed_value" | "generate" | "type"
   name: string
   function: string
   indexing_strategy: list string
   header: string

Column kind
^^^^^^^^^^^

The provided column kind tells the exporter how to handle elements in
the list originating from the serialization kind of the source
attribute. - Column kind ``type`` uses the ``indexing_strategy`` to
iteratively index the element and writes the type of the element as
values into the respective rows. - Column kind ``simple`` uses the
``indexing_strategy`` to iteratively index the element and writes the
output to the target table. - Column kind ``complex`` uses the
``indexing_strategy`` to index the individual elements and applies a
function to them - Column kind ``fixed_value`` writes a fixed value for
each element into the column of the target table. - Column kind
``generate`` identifies each individual unique value resulting from
indexing all elements and generates a new column for each of the unique
values. Subsequently, the rows are filled with the count that results
from counting the items in the element that matches the respective
column.

Name
^^^^

The column name that appears in the target table. Not needed for
``column_kind = "generate"``.

Function
~~~~~~~~

The function that is applied to each individual element when
``column_kind = "complex"``.

Indexing strategy
~~~~~~~~~~~~~~~~~

A list of keys that are iteratively used to index each individual
element, i.e. ``indexing_strategy = ["hello", "world"]`` the element
would be indexed as follows: ``el["hello"]["world"]``

Header
~~~~~~

An optional attribute that lets you specify whether columns should be
grouped. Columns with the same header are grouped together appearing
next to each other with an additional header in the target table.

Example:

.. code-block:: json

    {
      "target_schema": "xlsx",
      "sheets": [
        {
          "serialization_kind": "dict_flatten",
          "serialize_unique": true,
          "name": "StationaryResource",
          "source": "stationary_resources",
          "columns": [
            {
              "column_kind": "fixed_value",
              "name": "index",
              "value": "StationaryResource"
            }
          ]
        }
      ]
    }
