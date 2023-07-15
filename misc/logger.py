##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-08 5:54:10 pm
# @copyright MIT License
#

from __future__ import annotations

import logging
import os


class Logger:
    _instance = None
    _log_level_mapping = {"INFO": logging.INFO, "DEBUG": logging.DEBUG}

    def __new__(
        cls, log_path: str = None, log_level: str = "DEBUG", *args, **kwargs
    ) -> Logger:
        """Function to be called while creating the instance of this class.

        Args:
            log_path (str, optional): Path of file to save all logs. Defaults to None.
            log_level (str, optional): Log level of the logger. Defaults to "INFO".

        Returns:
            Logger: Logger singleton instance.
        """
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance = logging.getLogger("logger")

        cls._instance.setLevel(cls._log_level_mapping.get(log_level))
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s \t| %(filename)s:%(lineno)s] > %(message)s",
            datefmt="%d-%B-%Y %H:%M:%S",
        )

        if log_path is None:
            print("Log path is not provided.")
        else:
            if not os.path.isdir(log_path):
                os.mkdir(log_path)

            cls._logger_path = os.path.join(log_path, "training_logs.txt")
            file_handler = logging.FileHandler(cls._logger_path)
            file_handler.setFormatter(formatter)
            cls._instance.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        cls._instance.addHandler(stream_handler)

        return cls._instance


if __name__ == "__main__":
    logger = Logger("./log", "DEBUG")
    logger.info("Hello, Logger")
    logger = Logger()
    logger.debug("bug occured")
