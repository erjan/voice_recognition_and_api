import logging
import pandas as pd
import numpy as np

FORMAT = '[%(asctime)s] [%(process)d] [%(levelname)7s] [%(filename)s:%(lineno)s]: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S %z'


def init_logger(name):
    # Formatters
    formatter = logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT)

    # Handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

       # Loggers
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # root level
    logger.addHandler(console_handler)
    return logger


class ApiError(Exception):
    result = None
    errcode = None
    errtext = None

    def __init__(self, errcode, errtext):
        Exception.__init__(self)
        self.result = 'err'
        self.errcode = errcode
        self.errtext = errtext

    def get_error(self):
        return {'result': self.result, 'errcode': self.errcode, 'errtext': self.errtext}

    def __str__(self):
        return str(self.get_error())
