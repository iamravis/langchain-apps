import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import setup_logging

logger = setup_logging()

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    
    except Exception as e:
        raise CustomException(e)
 
 
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    ''' 
        This function takes train and test data 
        along with models and outputs the model
        score for the test data   
    '''
    
    try:
        report = {}       
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            logger.info(f"Fitting {i} - {model} ")
            y_train_pred = model.predict(X_train)
            
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        
        return report
            
    except Exception as e:
        raise CustomException(e)