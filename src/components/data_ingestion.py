import os
import sys
from src.exception import CustomException
from src.logger import setup_logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
logger = setup_logging()
class DataIngestion:
    ''' This class will save the data in train and test data'''
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):    
        logger.info("\033[91m 1. Entered the data ingestion method")
        
        try:
            # Read the raw data
            df = pd.read_csv('notebook/data/stud.csv')
            logger.info("\033[91m 2. Reading stud data successfully")
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Start train test split 
            logger.info('\033[91m 3. Train Test split started')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # save the train and test data 
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info("\033[91m 4. Data Ingestion is complete\n")
            
            # return train and test data
            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)
        
            
        except Exception as e:
            raise CustomException(e)
        
if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

