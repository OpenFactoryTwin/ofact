"""
contains the custom logger
"""
# Imports Part 1: Standard Imports
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import List
import os
# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.twin.settings import DEFAULT_LOGGER_NAME, CONSOLE_LOG_LEVEL, FILE_LOG_LEVEL, LOG_FOLDER, LOG_AMOUNT
# setup

# ---

# -------------------------------
# Logger
# -------------------------------
class CustomLogFormatter(logging.Formatter):
    """
    Custom Log Formatter for setting default formats and handling optional parameters [obj_id, behav_id].
    Will include the call-path of the logging-event.
    """
    COLORS = {
        'DEBUG': '\033[90m',  # Grau
        'INFO': '\033[97m',  # Weiß
        'WARNING': '\033[93m',  # Orange
        'ERROR': '\033[91m',  # Rot
        'CRITICAL': '\033[41m',  # Helles Rot als Hintergrund für kritische Fehler
        'RESET': '\033[0m'  # Reset der Farbe
    }

    def __init__(self, fmt: str = "{asctime} - {levelname} - {module_func} - {message}",
                 datefmt: str = "%Y-%m-%d %H:%M:%S", colorise: bool = False):
        """
        Initializes the custom log formatter with a default format and date format.
        :param fmt (str, optional): The log message format. Usually it does not need to be provided, a default format
         is used. If it is set, default is ignored, but optional parameters are still included into the message.
         Uses the {}-style formatting.
        :param datefmt (str, optional): The date format string. If not provided, includes seconds by default.
        """
        self.colorise = colorise
        # Set the default format if none is provided
        if fmt is None:
                fmt = "{asctime} - {levelname} - {module_func} - {message}"
        if datefmt is None:
            datefmt = "%Y-%m-%d %H:%M:%S"

        super().__init__(fmt, datefmt, style='{')

    def format(self, record):
        """
        Format the log record by adding additional information like object ID and behavior ID.

        :Param record (logging.LogRecord): The LogRecord instance that holds information about the event being logged.
        :return: The formatted log record.
        """
        # Handle object and behavior IDs
        record.obj_id = getattr(record, 'obj_id', '')
        record.behav_id = getattr(record, 'behav_id', '')

        # Adjust module/function formatting
        if record.funcName == '<module>':
            record.module_func = f'{record.module}'
        else:
            call_path =f'{record.module}'
            if record.obj_id:
                call_path += f'/{record.obj_id}'
            if record.behav_id:
                call_path += f'/{record.behav_id}'
            call_path += f'/{record.funcName}'
            record.module_func = call_path

        log_message = super().format(record)
        # If colorize is enabled, apply colors based on log level
        if self.colorise:
            levelname = record.levelname
            color = self.COLORS.get(levelname, self.COLORS['RESET'])
            log_message = f"{color}{log_message}{self.COLORS['RESET']}"

        return log_message

def setup_dual_logger(name:str = DEFAULT_LOGGER_NAME, file_log_level:int = FILE_LOG_LEVEL,
                 consol_log_level:int = CONSOLE_LOG_LEVEL):
    """
       A customized logger that differs from the default by
       - logging into console and into a file at the same time while a loglevel can be set individually
       - including a call-path into the log message (e.g. Filepath/FunctionName)
       - supports Object-based-messages with extra={'obj_id': self.agent.name}
       - supports Behaviour-based-messages with extra={'behav_id': self.__class__.__name__}
       - storing a limited amount of log-Files (currently 10 files that represent each one day of log events)

       Example Output:
       2025-03-10 14:30:00 - INFO - mymodule/transportAgent4/path_planning_behaviour/calculate_shortest_distance: msg

       How to use:
       setup the logger at the top level by
        from utils import setup_dual_logger  # the necessary import
        logger = setup_dual_logger()  # initial setup of the logger
        logger.info("normal log")  # logging non-object messages
       You can edit the loglevels, the path of the logfile etc. in the ofact.settings.py file.
       """

    logger = logging.getLogger(name)
    # Check if the logger already has handlers, if yes, return it (do not re-setup)
    if logger.handlers:
        return logger

    # define the lowest level of log messages that should be processed at all
    # this saves processing time when the log levels are set higher because fewer messages are loaded to memory
    logger.setLevel(min(file_log_level, consol_log_level))

    def custom_logfile_namer(default_name):
        # This will be called when doing the log rotation and it will improve the default naming-schema
        base_filename, ext, date = default_name.split(".")
        # limited editing possible at time of creation. e.g. if "_" instead of "." the deleting does no longer work.
        return f"{base_filename}.{date}.{ext}"

    # log to console: Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(consol_log_level)

    # Ensure the log folder exists, if not, create it
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)

    # log to file: TimedRotatingFileHandler to keep a limited amount of days of log files
    file_handler = TimedRotatingFileHandler(filename=LOG_FOLDER + 'log.log', when='midnight', interval=1,
                                            backupCount=LOG_AMOUNT)
    file_handler.setLevel(file_log_level)

    # edit the format of the logfile name
    file_handler.namer = custom_logfile_namer

    # customize the log-lines
    console_handler.setFormatter(CustomLogFormatter(colorise=True))
    file_handler.setFormatter(CustomLogFormatter(colorise=False))

    # apply both handlers (console and file) to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger