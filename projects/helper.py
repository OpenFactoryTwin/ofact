import os


def get_project_path(project_name: str):
    """Return the path to the project folder"""

    path = os.getcwd()
    if "projects" in path:
        project_path = path.split("projects")[0] + f"projects/{project_name}"
    else:
        project_path = f"/usr/src/app/projects/{project_name}"

    return project_path
