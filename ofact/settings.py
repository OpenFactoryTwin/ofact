#############################################################
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
#############################################################

import os
import platform

# Detect the current working directory
path = os.getcwd()

# Adjust ROOT_PATH based on the operating system and directory structure
if platform.system() == "Windows":
    # Ensure ROOT_PATH is set to "D:\ofact-intern\ofact" for Windows
    if "ofact" in path:
        ROOT_PATH = os.path.join(path.rsplit("ofact", 1)[0], "ofact", "ofact")
    else:
        raise NotImplementedError(f"Entrance point unknown: {path}")
else:
    # Linux-like paths (you can leave this as is if needed for other environments)
    if "/usr/src/app" not in path:
        if "projects" in path:
            ROOT_PATH = os.path.join(path.rsplit("projects", 1)[0], "ofact")
        elif "ofact" in path:
            ROOT_PATH = os.path.join(path.rsplit("ofact", 1)[0], "ofact")
        elif "factory_planner" in path:
            ROOT_PATH = os.path.join(path.rsplit("factory_planner", 1)[0], "ofact")
        else:
            raise NotImplementedError(f"Entrance point unknown: {path}")
    else:
        ROOT_PATH = "/usr/src/app/ofact"

# Ensure paths are absolute
ROOT_PATH = os.path.abspath(ROOT_PATH)
