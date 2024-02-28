# import logging
# import os
# from datetime import datetime

# # Directory for logs
# log_directory = os.path.join(os.getcwd(), "logs")
# os.makedirs(log_directory, exist_ok=True)

# # Dynamic log filename with the current timestamp
# log_filename = datetime.now().strftime('%d-%m-%Y_%H-%M-%S.log')
# log_file_path = os.path.join(log_directory, log_filename)

# logging.basicConfig(
#     filename=log_file_path,
#     format="[%(asctime)s]  %(lineno)d %(name)s  -  %(levelname)s  -  %(message)s",
#     level=logging.INFO
# )


    
import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging():
    # Directory for logs
    log_directory = os.path.join(os.getcwd(), "logs")
    # Ensure the directory exists
    if not os.path.exists(log_directory):
        os.makedirs(log_directory, exist_ok=True)

    # Base log filename
    log_filename = 'application.log'
    log_file_path = os.path.join(log_directory, log_filename)

    # Create a custom logger
    logger = logging.getLogger('MyAppLogger')
    logger.setLevel(logging.INFO)  # Adjust log level as needed

    # Define log format
    formatter = logging.Formatter("[%(asctime)s]  %(lineno)d %(name)s  -  %(levelname)s  -  %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    # Create TimedRotatingFileHandler for daily log rotation
    handler = logging.handlers.TimedRotatingFileHandler(
        log_file_path, when="midnight", interval=1, backupCount=30
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Optional: Add a stream handler to output logs to console (consider removing for production to reduce console clutter)
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    # Handle uncaught exceptions with the logging system
    # def handle_exception(exc_type, exc_value, exc_traceback):
    #     if issubclass(exc_type, KeyboardInterrupt):
    #         # Call the default sys.excepthook saved at start
    #         sys.__excepthook__(exc_type, exc_value, exc_traceback)
    #         return
    #     logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    # import sys
    # sys.excepthook = handle_exception

    return logger

