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

def get_attr_value(object_, attr_str):
    """Return the attribute value from the object_ that can be an object or a dict"""
    if isinstance(object_, dict):
        value = object_[attr_str]
    else:
        try:
            value = getattr(object_, attr_str)
        except:
            print("Attribute value problem:", object_, attr_str)
            value = getattr(object_, attr_str)

    return value
