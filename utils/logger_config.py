import logging
import os

def configure_logger(name, log_filename):
    """
    This function sets up a logger with name "name" and returns it.
    The logger logs messages both to the console and a log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Make sure we don't have duplicate handlers if executed more times
    if not logger.handlers:
        # Create logs directory if it doesn't exist
        script_dir = os.path.dirname(os.path.realpath(__file__))
        log_directory = os.path.join(script_dir, '../logs')
        os.makedirs(log_directory, exist_ok=True)

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(log_directory, log_filename), mode='w')
        fh.setLevel(logging.INFO)

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


