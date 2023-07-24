from datetime import datetime
import logging
import os


loggers = {}

def configure_logger(name):
    """
    This function sets up a logger with name "name" and returns it.
    The logger logs messages both to the console and a log file.
    If the logger already exists, it is returned as is.
    """
    if name in loggers:
        return loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create logs directory if it doesn't exist
    script_dir = os.path.dirname(os.path.realpath(__file__))
    log_directory = os.path.join(script_dir, '../logs')
    os.makedirs(log_directory, exist_ok=True)

    # Create file handler which logs even debug messages
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  # get the current timestamp
    log_filename = f"{timestamp}_{name}.log"  # prepend the timestamp to the filename
    fh = logging.FileHandler(os.path.join(log_directory, log_filename), mode='w')
    fh.setLevel(logging.INFO)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Store the logger for reuse
    loggers[name] = logger

    return logger



