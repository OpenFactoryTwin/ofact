# Create the project from scratch ...

https://www.jetbrains.com/help/pycharm/poetry.html

Open command prompt

> pip install poetry 
> 
> python.exe -m pip install --upgrade pip


Setting up toml file (manage the requirements)
> poetry init

Add a requirement (also appended to the toml file)
> poetry add <package name>
> 
> poetry remove <package name>

Create the virtual environment
If you want that the virtual env is stored in inside the project
> poetry config virtualenvs.in-project true

> poetry install

> poetry env info
> 
> poetry env into -p
> 
> Show all the envs:
> 
> poetry env list

> virtual envs can be deleted by delete the whole folder


Work with poetry:
Open the shell
> poetry shell
> 
> e.g.: poetry pytest
> 
> To get out of the evironment:
> 
> exit or deactivate