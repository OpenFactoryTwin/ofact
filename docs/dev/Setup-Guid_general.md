# Setup-Guide: Main-Project

Last edit: March 8, 2022

## Introduction

This setup-guide will help you to install a working development environment. To run a Project you will need also to
follow the project-specific setup-guide - depending on the Project you want to run.

Depending on your needs and your baseline situation you may not have to set up everything.
The Setup consists of the following steps:

- Setup of the Docker environment
    - You need this if you want to run the frontend locally on your device
    - You might need this for project specific services (e.g. XMPP in the MVP)
- Setup of the programming Environment
    - you need this to run, modify and extend the code

## 1. Setup Docker environment

### 1.1 Setup Docker

<aside>
❗ Docker Desktop for windows is only free for small business. Read more about this on their website.
Alternatively you could just run a linux-based system to avoid the licencing.

</aside>

<aside>
❗ while the installation-Setup you might need access to the BIOS of your Computer to enable virtualization

</aside>

go to their Website and download the installer:

```python
https://docs.docker.com/desktop/windows/install/
```

Install from the installer. While installation progress or after it, you might be prompted that you need to enable
Virtualization in the BIOS (if it is not set by default) and that you need to install WSL2 (if Linux Subshell
is not installed) For the second one there is a Download link provided.

Next it is recommended to create a **Docker-Folder** from where the project-images and runtime data will be saved.
The folder (depending on the needed containers) should look like this: 

![Untitled](development_with_git/imgs/yml_file.png)

<aside>
💡 if you want to run the full project you can copy the docker-compose from the project utilities folder

</aside>

### 1.2 Setup Frontend Container

For the Web-Frontend you need to prepare yourself: 

```python
https://dev.isst.fraunhofer.de/stash/projects/RIOTANAV2/repos/dashboard/browse?at=refs%2Fheads%2Ffeature%2FRIOTANAV2-76-frontend-auf-eine-api-vorbereiten
```

![Untitled](imgs/Untitled%201.png)

click on the download button as shown and save the files in a local directory.

<aside>
💡 currently the a Feature branch is used. but you may want to use the master-branch instead.

</aside>

Navigate into your Docker-Folder and edit the docker-compose.yml by adding a new service:

```python
[...]
services:
[...]
  dashboard:
    build:
      context: Dashboard-UI
      dockerfile: Dockerfile.local
    ports:
      - "8080:8080"
```

Now we will install the container-image once through the powershell - afterwards you can use buttons in the UI to get 
the containers up and running. Open the PowerShell by shift-right_click into the Docker-Folder and selecting 
“open PowerShell window here”. Into the powershell you type:

```python
docker-compose up
```

Now docker will build the changes (takes some time) and run the new service afterwards.

If the dashboard does not display very much you may need to configure some settings Main-Program (please refer to step 4).
Sometimes when Working with Windows an getaddrinfo EAI_AGAIN error appears. 
Hotfix ist to execute "ipconfig /renew" in powershell and build the container again with "docker build . --network host".

## 3. Setup of the Programming Environment

### 3.1 Setup Anaconda

Download and install Anaconda3

```markdown
[https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)
```

Run Conda-Prompt and install Python 3.9 environment named “RobotSwarm39” by using 

```markdown
conda create -n RobotSwarm39 python=3.9
```

### 3.2 Setup Pycharm

Next download and install Python IDE (PyCharm Pro Used [free for students], Community version and alternate IDE might work as well)

```markdown
https://www.jetbrains.com/de-de/pycharm/download/#section=windows
```

Checkout Project from VCS (You need access to it, alternative you may want to ask for a copy to work on it)

```markdown
https://dev.isst.fraunhofer.de/stash/scm/riotanav2/digitaltwin.git
```

Set Python Interpreter to “Existing environment” and use “RobotSwarm39” (the one you created beforehand)

Recommended: Pycharm does not allow for Program interrupts by default, but you might want to allow a clean close of the
Program while implementing/debugging (not the stop button - that acts like a kill).
In the Run/debug Configuration you can set the terminal to interpret interrupt

![Untitled](imgs/Untitled%202.png)

### 3.3 Install required dependencies

The Project does use several dependencies which need to be installed.

No matter what project you are working on you need to run

```python
pip install -r requirements.txt
```

to install the dependencies that are used by all projects.

Then you need to install the project-specific dependencies too. Example: To install the MVP-Project dependencies you
need to run the following:

```python
pip install -r projects\MVP\requirements.txt
```

<aside>
❗ If you have problems with a dependency they may have changed something relevant. In this case you can go into the
requirements file and uncomment the line with a approved version of the dependency instead of downloading the latest.
But keep in mind that this is more a “hotfix” than a solution!

</aside>
