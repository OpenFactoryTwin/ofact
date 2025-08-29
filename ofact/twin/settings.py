"""
Settings
"""
# Imports Part 1: Standard Imports
import os
import logging
# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
# setup
# ---

ROOT_PATH = os.getcwd()
# LOG default settings
DEFAULT_LOGGER_NAME = "dual_logger"

# possibilities:
#  CRITICAL
#  ERROR
#  WARNING
#  INFO
#  DEBUG
#  NOTSET

CONSOLE_LOG_LEVEL = logging.CRITICAL  # defines which messages should be displayed in the console
FILE_LOG_LEVEL = logging.CRITICAL  # defines which messages should be displayed in the Log-File
LOG_FOLDER = ROOT_PATH + "/logs/" # defines where the logfiles should are stored
LOG_AMOUNT = 10  # how many logfiles should be kept (starts a new file every day)
