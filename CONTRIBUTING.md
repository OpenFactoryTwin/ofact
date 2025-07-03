# Contributing to the Project

Thank you for your interest in contributing to the Open Factory Twin!

## Table of contents

* [Issues and Pull Requests](#issues-and-pull-requests)
* [License Header](#license-header)
* [Open Project in IntelliJ](#open-project-in-pycharm)
* [Style Guide](#style-guide)
    * [Auto formatter](#auto-formatter)
    * [Git](#Git)

## Issues and Pull Requests

To organize and structure all contributions, open an issue for enhancements, feature requests, or
bugs. In case of security issues, please see `SECURITY.md` for more details.

Code changes are handled entirely over pull requests. We use
the [squash and merge](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-commits)
strategy. Therefore, remember to use a pull request title in the
[conventional commits](https://www.conventionalcommits.org/) format since it is used as the squashed
commit message.

## Open Project in PyCharm

To work on all subprojects from one PyCharm, instance, you may have to manually create the env by installing 
the packages from `pyprojects.toml` or from the `requirements.txt` file.
Depending on the code area, you are working on not all packages are required. To identify all required packages 
for your use case, read the comments in the requirements/ pyprojects files.

## Style Guide

We aim for a coherent and consistent code base, thus the coding style detailed here should be
followed.
Therefore, we follow the [PEP 8](https://peps.python.org/pep-0008/) style guide for python code.

### Auto formatter

To ensure that these style guides are followed, we advise using an auto formatter.

### Git

We follow the [conventional commit guidelines](https://www.conventionalcommits.org).
