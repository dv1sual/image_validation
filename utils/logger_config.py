import os
import pathlib
import logging
from datetime import datetime
import json


def configure_logger(name, caller_dir):
    """
    This function sets up a logger with name "name" and returns it.
    The logger logs messages both to the console and a log file.
    If the logger already exists, it is returned as is.
    """
    # Load the logging configuration
    with open('config/logging_config.json', 'r') as f:
        log_config = json.load(f)

    logger = logging.getLogger(name)

    if not logger.handlers:  # No handlers, means we can create them
        logger.setLevel(log_config['log_level'])

        # Create logs directory if it doesn't exist
        log_directory = os.path.join(caller_dir, 'logs')
        os.makedirs(log_directory, exist_ok=True)

        # Create file handler which logs even debug messages
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  # get the current timestamp without milliseconds
        log_filename = f"{timestamp}_{name}.log"  # prepend the timestamp to the filename
        fh = logging.FileHandler(os.path.join(log_directory, log_filename), mode='w')
        fh.setLevel(log_config['log_level'])

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(log_config['log_level'])

        # Create formatter and add it to the handlers
        formatter = logging.Formatter(log_config['log_format'], datefmt=log_config['date_format'])
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
