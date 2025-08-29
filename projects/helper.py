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
from pathlib import Path


def get_project_path(project_name: str):
    """Return the path to the project folder"""

    path = os.getcwd()  # alternative required
    if "projects" in path:
        project_path = str(Path(path.split("projects")[0]) / "projects" / project_name) # str(Path(path.split("projects")[0] + f"/projects/{project_name}"))
    else:
        project_path = str(Path(path.split("ofact")[0]) / "projects" / project_name)
    return project_path
