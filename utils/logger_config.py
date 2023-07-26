import os
import pathlib
import logging
from datetime import datetime


def configure_logger(name):
    """
    This function sets up a logger with name "name" and returns it.
    The logger logs messages both to the console and a log file.
    If the logger already exists, it is returned as is.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create logs directory if it doesn't exist
    script_dir = pathlib.Path(__file__).parent.absolute()
    log_directory = os.path.join(script_dir, 'logs')
    os.makedirs(log_directory, exist_ok=True)

    # Create file handler which logs even debug messages
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  # get the current timestamp without milliseconds
    log_filename = f"{timestamp}_{name}.log"  # prepend the timestamp to the filename
    fh = logging.FileHandler(os.path.join(log_directory, log_filename), mode='w')
    fh.setLevel(logging.DEBUG)  # Change this to DEBUG

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # Change this to DEBUG

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

