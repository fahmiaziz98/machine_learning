import numpy as np
from typing import Dict, Union
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor


def create_enet_pipeline(
    params: Dict[str, Union[float, int]]
) -> "Pipeline[RobustScaler, ElasticNet]":
    """
    Create and return an ElasticNet Regression pipeline with RobustScaler as its first step.

    Args:
        params (Dict[str, Union[float, int]]): hyperparameters for ElasticNetRegressor

    Returns:
        Pipeline[RobustScaler, ElasticNet]: A pipeline with RobustScaler and ElasticNetRegressor
    """
    pipeline = make_pipeline(
        RobustScaler(), ElasticNet(**params)  
    )
    return pipeline



def create_rf_pipeline(
    params: Dict[str, Union[int, float]]
) -> "Pipeline[RobustScaler, RandomForestRegressor]":
    """
    Create and return a Random Forest Regression pipeline with RobustScaler as its first step

    Args:
        params (Dict[str, Union[int, float]]): hyperparameters for RandomForestRegressor

    Returns:
        Pipeline[RobustScaler, RandomForestRegressor]: A pipeline with RobustScaler and RandomForestRegressor
    """
    pipeline = make_pipeline(
        RobustScaler(), RandomForestRegressor(**params)
    )
    return pipeline



class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    Averages predictions of multiple base models.

    This class can be used to average predictions of multiple base models. The class
    implements the scikit-learn API and can be used as a normal estimator.
    """

    def __init__(self, models):
        """
        Initialize the AveragingModels class.

        Parameters:
            models: List of base models to average predictions from.
        """
        self.models = models

    def fit(self, X, y):
        """
        Fit all base models with the given data.

        Parameters:
            X: Feature data.
            y: Target data.
        """
        self.models_ = [clone(model) for model in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        """
        Average predictions of multiple base models.

        Parameters:
            X: Feature data.

        Returns:
            np.ndarray: Averaged predictions.
        """
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)
