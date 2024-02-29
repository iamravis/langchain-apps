import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.utils import logger
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        
        ''' This function is responsible for data transformation'''
        
        try:
            # Filter numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender",
                                    "race_ethnicity",
                                    "parental_level_of_education",
                                    "lunch",
                                    "test_preparation_course"]
            
            # Transformation pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scalar", StandardScaler())
                ]
            )
            
            # Transformation pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scalar", StandardScaler(with_mean=False))
                ]
            )
            
            logger.info(f"2. Defined categorical and numerical columns pipeline")
            
            # Proprcess the data using numerical and categorical pipeline defined above
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
            
            
        except Exception as e:
            raise CustomException(e)
        
    
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info("1. Read train and test data. Starting preprocessing..")
            
            # data transformation object 
            preprocess = self.get_data_transformer_object()
            
            # Define the target and 
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]
            
            logger.info(f"3. Applying the preprocessing on train and test data")
            
            input_feature_train_arr = preprocess.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocess.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)   
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path, obj=preprocess)
            
            logger.info(f"4. Data transformation step is complete and preprocessor.pkl file is ready\n")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e)    