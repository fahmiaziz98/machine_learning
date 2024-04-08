import warnings
import mlflow
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union
from pathlib import Path

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from mlflow_scikit.preprocces import preprocces_data
from mlflow_scikit.metrics import rmsle_cv, rmsle
from mlflow_scikit.base import create_enet_pipeline, create_rf_pipeline, AveragingModels
from mlflow_scikit.utils import logger

# Ignore all warnings
warnings.simplefilter("ignore")



def train_with_mlflow(
    train: pd.DataFrame, test: pd.DataFrame,
    y_train: np.ndarray, path: Path[str]
) -> None:
    """
    Train the model using mlflow tracking

    Args:
        train (pd.DataFrame): Training data
        test (pd.DataFrame): Testing data
        y_train (np.ndarray): True target values for training data
        path (Path[str]): Path to the data files

    Returns:
        Tuple[float, float, float]: RMSLE, MAE, MAPE
    """
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Check if the experiment already exists
    experiment_name = "House Price Prediction"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as run:
        raw_data = pd.read_csv(f"{path}/train.csv")
        # Create an instance of a PandasDataset
        # https://mlflow.org/docs/latest/tracking/data-api.html
        dataset = mlflow.data.from_pandas(
            raw_data, source=f"{path}/train.csv", name="House Rental"
        )
        mlflow.log_input(dataset, context="training")

        client = MlflowClient()
        
        # Define hyperparameters for ElasticNetRegressor
        params_enet: Dict[str, float] = {
            "alpha": 0.0005,  
            "l1_ratio": 0.9,  
            "random_state": 101  
        }
        params_rf: Dict[str, Union[int, float]] = {
            "n_estimators": 100,  
            "max_depth": 5,  
            "max_features": 0.5,  
            "min_samples_leaf": 10,  
            "random_state": 101  
        }

        logger.info("Training...")
        ENet = create_enet_pipeline(params_enet)
        RFRegressor = create_rf_pipeline(params_rf)
        averaged_models = AveragingModels(models=[ENet, RFRegressor])

        # Log hyperparameters and set a tag
        mlflow.log_params({**params_enet, **params_rf})
        mlflow.set_tag("Training Info", "StackModel Regressor (Enet RFRegressor)")

        rmse_cv, mae, mape = rmsle_cv(model=averaged_models, train=train, y_train=y_train)
        logger.info(f"Model is trained.\nRMSE : {rmse_cv}\nMAE: {mae}\nMAPE: {mape}")

        mlflow.log_metric("rmse_cv", rmse_cv)
        mlflow.log_metric("mae_cv", mae)
        mlflow.log_metric("mape_cv", mape)

        logger.info("Train on Train set...")
        averaged_models.fit(train, y_train)
        # preds = averaged_models.predict(test.values)
        train_preds = averaged_models.predict(train.values)
        rmse = rmsle(y_train, train_preds)
        logger.info(f"Root Mean Squared Log Error on Train set: {rmse}")
        mlflow.log_metric("rmse", rmse)
        
        signature = infer_signature(test, averaged_models.predict(train))
        mlflow.sklearn.log_model(
            sk_model=averaged_models,
            artifact_path="averaged_models",
            signature=signature,
            input_example=train,
            registered_model_name="StackModel"
        )
        logger.info("Model saved...")

        
if __name__ == "__main__":

    PATH = "/home/fahmiaziz/project_py/machine_learning/mlflow-scikit/data"

    train, test, y_train = preprocces_data(path=PATH)
    train_with_mlflow(train, test, y_train, path=PATH)