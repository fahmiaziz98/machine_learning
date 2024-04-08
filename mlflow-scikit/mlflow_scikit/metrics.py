import numpy as np
from typing import Tuple
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def eval_and_log_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Evaluate metrics 

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values

    Returns:
        Tuple[float, float]: MAE, MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return mae, mape



def rmsle(
    y: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Compute the RMSLE (root mean squared log error) between two arrays.

    Args:
        y (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values

    Returns:
        float: RMSLE value
    """
    return np.sqrt(mean_squared_error(y, y_pred))

n_folds = 5

def rmsle_cv(
    model: BaseEstimator, train: np.ndarray, y_train: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute RMSLE, MAE, MAPE using cross-validation

    Args:
        model (BaseEstimator): Trained model
        train (np.ndarray): Feature matrix
        y_train (np.ndarray): True target values

    Returns:
        Tuple[float, float, float, float]: RMSLE, MAE, MAPE
    """
    rmse = cross_val_score(
        model, train, y_train, cv=KFold(n_folds, shuffle=True, random_state=42), scoring="neg_mean_squared_error"
    )
    rmse = np.sqrt(-np.mean(rmse))
    y_pred = cross_val_predict(model, train, y_train, cv=KFold(n_folds, shuffle=True, random_state=42))
    mae, mape = eval_and_log_metrics(y_train, y_pred)
    
    return rmse, mae, mape




