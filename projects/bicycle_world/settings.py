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

contains only constants and global settings for the MVP
@author:Roman Sliwinski
@version:2021.10.12
"""

# Imports 3: Project Imports
from projects.helper import get_project_path

# no Imports

# constants and settings

PROJECT_NAME = "bicycle_world"
PROJECT_PATH = get_project_path(PROJECT_NAME)
print("Project Path:", PROJECT_PATH)

# localhost
localhost = "127.0.0.1"
# localhost = "10.0.2.2"

# XMPP
XMPP_SERVER_IP_ADDRESS = localhost  # The URL of where the Openfire-XMPP Server is running - usually Localhost if it runs on the same machine
XMPP_SERVER_SHARED_SECRET = "89Su4JA1ep8XmHkw"  # The shared Secret Key of the Openfire-REST API
XMPP_ALL_AGENTS_PASSWORD = "LCtBjPge9y6fCyjb"  # in the MVP all Agents share the same password
XMPP_SERVER_REST_API_USERS_ENDPOINT = "/plugins/restapi/v1/users"  # see https://github.com/seamus-45/openfire-restapi/blob/master/docs/users.md
XMPP_SERVER_REST_API_PORT = ":9090"

# Vector
ROBOT_CONNECTION_RETRIES = 5  # how many times a reconnect is tried out before aborting
ROBOT_CONNECTION_TIMEOUT = 12  # how long each try is at max before aborting it
ROBOT_SPEED_TURN = 50  # [10..100] degrees per second
ROBOT_SPEED_STRAIGHT = 150  # [20..200] the speed the robot drives
ROBOTS = {"x4v4": {"serial": "00404205", "async": True},
          "k8j9": {"serial": "00701643", "async": True}}

# Shop Floor
ROBOTS_USE_REAL = False

# API
API_HOST = "localhost"
API_PORT = 3000

API_GET_ORDERS_SCHEME = {
    "dateStart": {"required": True, "type": "integer"},
    "dateEnd": {"required": True, "type": "integer"},
    "orders": {"required": True, "type": "list"},
    "products": {"required": True, "type": "list"},
    "processes": {"required": True, "type": "list"},
    "resources": {"required": True, "type": "list"}
}

API_GET_PRODUCTS_SCHEME = {
    "dateStart": {"required": True, "type": "integer"},
    "dateEnd": {"required": True, "type": "integer"},
    "orders": {"required": True, "type": "list"},
    "products": {"required": True, "type": "list"},
    "processes": {"required": True, "type": "list"},
    "resources": {"required": True, "type": "list"}
}

API_GET_PROCESSES_SCHEME = {
    "dateStart": {"required": True, "type": "integer"},
    "dateEnd": {"required": True, "type": "integer"},
    "orders": {"required": True, "type": "list"},
    "products": {"required": True, "type": "list"},
    "processes": {"required": True, "type": "list"},
    "resources": {"required": True, "type": "list"}
}

API_GET_RESOURCES_SCHEME = {
    "dateStart": {"required": True, "type": "integer"},
    "dateEnd": {"required": True, "type": "integer"},
    "orders": {"required": True, "type": "list"},
    "products": {"required": True, "type": "list"},
    "processes": {"required": True, "type": "list"},
    "resources": {"required": True, "type": "list"}
}

language = "DE"  # also "EN" possible
