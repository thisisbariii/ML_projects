import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            # Separate input features (X) and target/output (y)
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # All models we want to test
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params ={
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                },
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    'n_neighbors':[5,7,9,11],
                },
                "XGB Regressor":{
                    'learning_rate':[.1,0.1,0.5,.001],
                    'n_estimators':[8,16,32,64,128,256]

                },
                "CatBoosting Regressor":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,50,100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators':[8,16,32,64,128,256,]
                }
            }
            # Evaluate all models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params

            )

            print("Model Report:", model_report)

            # Get the highest model score
            best_model_score = max(model_report.values())

            # Get the name of the model with the highest score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Get the actual best model
            best_model = models[best_model_name]

            print("Best Model:", best_model_name)
            print("Best Score:", best_model_score)

            # Check if model is good enough
            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found with acceptable score",
                    sys
                )

            logging.info(
                f"Best model found: {best_model_name} "
                f"with score: {best_model_score}"
            )

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print(
                f"Model saved successfully at: "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            # Predict using best model
            predicted = best_model.predict(X_test)

            # Calculate final R2 score
            final_r2_score = r2_score(y_test, predicted)

            logging.info(
                f"Final R2 Score: {final_r2_score}"
            )

            # Return model score
            return final_r2_score

        except Exception as e:
            raise CustomException(e, sys)