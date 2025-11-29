import os
from datetime import datetime
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        
    def log(self, message):
        print(message)  # Still print to console
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()}: {message}\n")
