###############################################################
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
###############################################################

from ofact.twin.repository_services.store_digital_twin_model import store_state_model
from ofact.planning_services.model_generation.twin_generator import get_digital_twin
from ofact.settings import ROOT_PATH
from projects.bicycle_world.settings import PROJECT_PATH

if __name__ == "__main__":
    digital_twin_file_name = "base_wo_material_supply.xlsx"  # "base.xlsx"
    state_model = get_digital_twin(PROJECT_PATH, digital_twin_file_name, pickle_=False,
                                        customer_generation_from_excel=True, order_generation_from_excel=True,
                                        customer_amount=10, order_amount=10,)

    store_ = False
    if store_:
        root_path = ROOT_PATH
        digital_twin_result_pkl = "models/twin/base_wo_material_supply.pkl"
        store_state_model(state_model, digital_twin_result_path=digital_twin_result_pkl)
