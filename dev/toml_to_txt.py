"""
Convert pyproject.toml to requirements.txt
"""

import toml
from pathlib import Path

# Load the pyproject.toml file
pyproject_path = Path('../../pyproject.toml')
requirements_path = Path('../../requirements.txt')

# Read the pyproject.toml file
pyproject_data = toml.load(pyproject_path)

# Extract dependencies
dependencies = pyproject_data.get('tool', {}).get('poetry', {}).get('dependencies', {})

# Write to requirements.txt
with open(requirements_path, 'w') as req_file:
    for package, version in dependencies.items():
        if package != 'python':  # Skip python version
            # Format version properly
            if isinstance(version, str):
                req_file.write(f"{package}{version.replace('^', '==')}\n")

print(f"Requirements have been written to {requirements_path}.")