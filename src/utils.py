import os
import sys
import dill
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        if obj is None:
            raise CustomException("Attempted to save a None object", sys)

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(xtrain, ytrain, xtest, ytest, models):
    report = {}

    for model_name, model in models.items():
        model.fit(xtrain, ytrain)
        ytest_pred = model.predict(xtest)
        r2 = r2_score(ytest, ytest_pred)
        report[model_name] = r2

    return report


def model_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def print_evaluated_results(xtrain, ytrain, xtest, ytest, model):
    model.fit(xtrain, ytrain)
    ytrain_pred = model.predict(xtrain)
    ytest_pred = model.predict(xtest)

    print("Training R2 Score :", r2_score(ytrain, ytrain_pred))
    print("Test R2 Score     :", r2_score(ytest, ytest_pred))
