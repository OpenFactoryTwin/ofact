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

Used to export the (dynamic) digital twin state model to different persistence formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ofact.twin.repository_services.serialization.state_model import StateModelSerialization
from ofact.twin.state_model.seralized_model import SerializedStateModel
try:
    from ofact.twin.repository_services.db.mongodb import StateModelDB
except ModuleNotFoundError:
    pass

try:
    import gridfs
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel


class DynamicStateModelSerialization(StateModelSerialization):
    # Note: dynamic attributes of the physical body are not considered

    def __init__(self,
                 state_model: StateModel,
                 mapping_file: str = "./dynamic_model_mapping.json",
                 db: Optional[StateModelDB] = None):
        super().__init__(state_model=state_model, mapping_file=mapping_file)

        self.db = db

    def _export_pkl(self, target_file: str):
        if "sources" not in self.mapping.keys():
            raise KeyError("No sources specified in mapping file.")

        SerializedStateModel.dump_to_pickle(self.serialized_state_model_dict, target_file)

    def _export_json(self, target_file: str):
        """
        Export the serialized state model to a JSON file.

        Parameters
        ----------
        target_file : str
            The path to the target JSON file.

        Returns
        -------
        None
        """
        if "sources" not in self.mapping.keys():
            raise KeyError("No sources specified in mapping file.")

        SerializedStateModel.dump_to_json(self.serialized_state_model_dict, target_file)

    def _export_parquet(self, target_file: str):
        SerializedStateModel.dump_to_parquet_folder(self.serialized_state_model_dict, target_file)

    def _export_db(self, *args):

        """chunk_size = 100000  # Größe der Teile (Anzahl der Einträge pro Teil)
        chunks = [self.serialized_state_model[i:i + chunk_size] for i in range(0, len(self.serialized_state_model), chunk_size)]
        for i,chunk in enumerate(chunks):
            self.db.persist({'chunk_id': i, 'data': chunk})
        """
        state_model_json = SerializedStateModel.to_json(self.serialized_state_model_dict)
        # fs = gridfs.GridFS(self.db.db)
        # file_id = fs.put(state_model_json.encode('utf-8'))
        # print(f'Upload ID ist : {file_id}')

if __name__ == "__main__":
    from pathlib import Path
    from ofact.settings import ROOT_PATH
    from datetime import datetime
    from ofact.twin.repository_services.db.mongodb import StateModelMongoDB

    print(datetime.now())
    state_model_path = Path(ROOT_PATH.split("ofact")[0],
                                        "ofact/twin/repository_services/serialization/test.pkl")
    state_model = None
    print("State Model loaded", datetime.now())
    exporter = DynamicStateModelSerialization(state_model=state_model, db=None)  # StateModelMongoDB())
    print("State Model serialized", datetime.now())
    exporter.export("test_result.pkl")
    print("State Model stored again", datetime.now())
