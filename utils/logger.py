import sys
import os
import logging
import time
from config import root_path
class Logger(object):

    def __init__(self, filename):
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        fileHandler = logging.FileHandler(filename)
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(formatter)

        # show on
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setLevel(logging.DEBUG)
        consoleHandler.setFormatter(formatter)

        # add to Handler
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(consoleHandler)

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()

def create_logger(args):
    timestamp = time.time()
    current_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(timestamp))
    model_name = args['base_model']
    data_name = args['dataset_name']
    run_id = f"{timestamp:.8f}"
    log_dir = os.path.join(root_path , "log", data_name, model_name)
    logger_name = os.path.join(log_dir, current_time_str + ".log")
    logger = Logger(logger_name)

    logger.info(f"my pid: {os.getpid()}")
    logger.info(args)

    return logger