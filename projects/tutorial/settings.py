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
from projects.helper import get_project_path

PROJECT_NAME = "tutorial"
PROJECT_PATH = get_project_path(PROJECT_NAME)

# localhost
localhost = "127.0.0.1"
# localhost = "10.0.2.2"

# XMPP
XMPP_SERVER_IP_ADDRESS = localhost  # The URL of where the Openfire-XMPP Server is running - usually Localhost if it runs on the same machine
XMPP_SERVER_SHARED_SECRET = "89Su4JA1ep8XmHkw"  # The shared Secret Key of the Openfire-REST API
XMPP_ALL_AGENTS_PASSWORD = "LCtBjPge9y6fCyjb"  # in the MVP all Agents share the same password
XMPP_SERVER_REST_API_USERS_ENDPOINT = "/plugins/restapi/v1/users"  # see https://github.com/seamus-45/openfire-restapi/blob/master/docs/users.md
XMPP_SERVER_REST_API_PORT = ":9090"
