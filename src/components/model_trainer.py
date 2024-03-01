import os
import sys

from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from src.exception import CustomException
from src.logger import setup_logging
from src.utils import save_object, evaluate_models
from src.utils import logger

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("1. Splitting training and test input data")
            
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Classifier" : KNeighborsRegressor(),
                "XGBClassifier" : XGBRegressor(),
                "CatBoosting Classifier" :CatBoostRegressor(verbose=False),
                "AdaBoost Classifier" : AdaBoostRegressor()
            }
            
            model_report: dict = evaluate_models(X_train = X_train, 
                                                 y_train = y_train, 
                                                 X_test = X_test,
                                                 y_test = y_test,
                                                 models = models)
            logger.info('2. Models evaluation is complete')
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model = models[best_model_name]
            
            
            if best_model_score <0.6:
                raise CustomException("None of the models are performing well.")
            
            logger.info("3. Models training phase is complete")
            logger.info(f'4. Best model selected is: {best_model_name} with score: {best_model_score}')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            return r2_square
            
        except Exception as e:
            raise CustomException