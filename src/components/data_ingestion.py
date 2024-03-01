import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.utils import logger
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer  import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    

class DataIngestion:
    ''' This class will save the data in train and test data'''
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    
    def initiate_data_ingestion(self):    
        logger.info("1. Entered the data ingestion method")
        
        try:
            # Read the raw data
            df = pd.read_csv('notebook/data/stud.csv')
            logger.info("2. Reading stud data successfully")
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Start train test split 
            logger.info('3. Train Test split started')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # save the train and test data 
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info("4. Data Ingestion is complete\n")
            
            # return train and test data
            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)
        
            
        except Exception as e:
            raise CustomException(e)
        
if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_array, test_array = data_transformation.initiate_data_transformation(train_data, test_data)
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array, test_array))

