import sys
from src.logger import setup_logging



def error_message_detail(error, exc_tb):
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = "\n\n\033[91mError occurred in file :\t{0} \n\nLine number:\t{1} \n\nError Message :\t{2} \n".format(filename, exc_tb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        exc_type, exc_value, exc_tb = sys.exc_info()
        self.error_message = error_message_detail(error_message, exc_tb)
        
    def __str__(self):
        return self.error_message

# logger = setup_logging()

# if __name__ == "__main__":
#     try:
#         a=1/0
#     except Exception as e:
#         logger.info("Divided by Zero")
#         raise CustomException(e)
        