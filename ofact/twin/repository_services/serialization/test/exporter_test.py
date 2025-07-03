"""
#############################################################
This program and the accompanying materials are made available under the
terms of the Apache License, Version 2.0 which is available at
https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

SPDX-License-Identifier: Apache-2.0
#############################################################
"""

import os
import unittest
from pathlib import Path
from time import sleep

import pandas as pd

from ofact.planning_services.model_generation.static_state_model_generator import StaticStateModelGenerator
from ofact.twin.repository_services.serialization.state_model import StateModelSerialization
from ofact.twin.repository_services.deserialization.order_types import OrderType


class MyTestCase(unittest.TestCase):
    def test_excel_export(self):
        """This test tries to import export import export
        This shows several mistakes, which wouldn't be found if only one export is tried."""
        twin = StaticStateModelGenerator(r"files/mini.xlsx",
                                         CUSTOMER_GENERATION_FROM_EXCEL=True,
                                         ORDER_GENERATION_FROM_EXCEL=True,
                                         ORDER_TYPE=OrderType.SHOPPING_BASKET
                                         ).get_state_model()

        exporter = StateModelSerialization(twin)
        exporter.export("files/test_identical_export.xlsx")
        twin = StaticStateModelGenerator("files/test_identical_export.xlsx",
                                         CUSTOMER_GENERATION_FROM_EXCEL=True,
                                         ORDER_GENERATION_FROM_EXCEL=True,
                                         ORDER_TYPE=OrderType.SHOPPING_BASKET
                                         ).get_state_model()

        exporter = StateModelSerialization(twin)
        exporter.export("files/test_identical_export.xlsx")

    def test_import_export_identical(self):
        """This imports a file and exports it again. Then the import and export are compared. """

        twin = StaticStateModelGenerator(r"./test/test_identical_export.xlsx",
                                         CUSTOMER_GENERATION_FROM_EXCEL=True,
                                         ORDER_GENERATION_FROM_EXCEL=True,
                                         ORDER_TYPE=OrderType.SHOPPING_BASKET
                                         ).get_state_model()

        exporter = StateModelSerialization(twin)
        exporter.export("./test_export.xlsx")

        # Import all sheets from the original file
        xlsx = pd.ExcelFile(r"./interface/test/test_identical_export.xlsx")
        sheets_orig = {}
        for sheet in xlsx.sheet_names:
            sheets_orig[sheet] = xlsx.parse(sheet)
        xlsx.close()
        # Import all sheets from the exported file
        xlsx = pd.ExcelFile(r"./test_export.xlsx")
        sheets_export = {}
        for sheet in xlsx.sheet_names:
            sheets_export[sheet] = xlsx.parse(sheet)

        xlsx.close()
        for sheet_orig, sheet_export in zip(sheets_orig, sheets_export):

            if sheet_orig == "Orders":
                # The features for orders are not in the same order, so we have to sort them first by using them
                # as index
                sheets_orig[sheet_orig] = sheets_orig[sheet_orig].T.sort_values(sheets_orig[sheet_orig].index[0],
                                                                                ascending=True).T
                sheets_export[sheet_export] = sheets_export[sheet_export].T.sort_values(
                    sheets_export[sheet_export].index[0], ascending=True).T
                sheets_export[sheet_export].columns = sheets_orig[sheet_orig].columns

            df_diff = pd.concat([sheets_orig[sheet_orig], sheets_export[sheet_export]],
                                ignore_index=True).drop_duplicates(keep=False)
            self.assertTrue(df_diff.empty,
                            msg=f"The sheets are not identical (sheet orig and sheet export, differences):"
                                f"{sheet_orig},{sheet_export},\n {df_diff}")

    @classmethod
    def tearDownClass(cls):

        sleep(1)  # Wait a short moment to allow the file to be closed completely
        if os.path.exists("./test_export.xlsx"):
            try:
                os.remove("./test_export.xlsx")
            except:
                print("File could not be deleted. Maybe it is still open?")
        if os.path.exists("files/test_identical_export.xlsx"):
            os.remove("files/test_identical_export.xlsx")


if __name__ == '__main__':
    unittest.main()
