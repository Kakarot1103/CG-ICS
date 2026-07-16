import logging

def setup_logger(log_file=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console log handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File log handler (only saves logs to file on rank 0)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    return logger
