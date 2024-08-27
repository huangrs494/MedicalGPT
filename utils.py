import logging
import re
import os
from logging.handlers import TimedRotatingFileHandler

def setup_log(log_name):
    # create logger instance. input logger name.

    logger = logging.getLogger(log_name)
    # log_path = os.path.join("moss/log", log_name) # DOCKER
    log_path = os.path.join("logs", log_name)  # NOHUP
    # set the level of log
    logger.setLevel(logging.INFO)
    # interval rotating period.
    # when="MIDNIGHT", interval=1 update at 00:00:00, create one file every day.
    # backupCount  represents the bum of logger files to save.
    # file_handler = TimedRotatingFileHandler(
    #     filename=log_path, when="MIDNIGHT", interval=1, backupCount=2
    # )
    file_handler = TimedRotatingFileHandler(
        filename=log_path, when="MIDNIGHT", interval=1, backupCount=30
    )
    # filename="mylog" set suffix and will create file name like mylog.2020-02-25.log
    file_handler.suffix = "%Y-%m-%d.log"
    # extMatch是编译好正则表达式，用于匹配日志文件名后缀
    # 需要注意的是suffix和extMatch一定要匹配的上，如果不匹配，过期日志不会被删除。
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
    # define the output format of logger
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s"
        )
    )
    logger.addHandler(file_handler)
    return logger
