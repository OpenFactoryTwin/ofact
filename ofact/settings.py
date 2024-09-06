"""The standard settings are throughout the whole repository and can be overwritten in the project contexts"""
import os

path = os.getcwd()
if "/usr/src/app" not in path:
    if "projects" in path:
        ROOT_PATH = path.split("projects")[0] + "ofact"
    elif "ofact" in path:
        ROOT_PATH = path.split("ofact")[0] + "ofact"
    else:
        raise NotImplementedError("Entrance point unknown: ", path)

else:
    ROOT_PATH = "/usr/src/app/ofact"
    # raise Exception("Root path did not contain the necessary DigitalTwin folder as entrance point for the whole repo: ", path)
