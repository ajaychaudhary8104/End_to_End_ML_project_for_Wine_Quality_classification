import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from src.mlProject.entity.config_entity import ModelEvaluationConfig
from src.mlProject.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    


    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        import dagshub
        dagshub.init(repo_owner='ajaychaudhary8104', repo_name='End_to_End_ML_project_for_Wine_Quality_classification', mlflow=True)
        try:
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():

                predicted_qualities = model.predict(test_x)

                (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
                
                # Saving metrics as local
                scores = {"rmse": rmse, "mae": mae, "r2": r2}
                save_json(path=Path(self.config.metric_file_name), data=scores)

                mlflow.log_params(self.config.all_params)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    try:
                        # Register the model
                        # There are other ways to use the Model Registry, which depends on the use case,
                        # please refer to the doc for more information:
                        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                        mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
                    except Exception as e:
                        print(f"Warning: Could not register model in MLflow: {str(e)}")
                        print("Logging model without registration...")
                        mlflow.sklearn.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
                    
        except Exception as e:
            print(f"Warning: MLflow logging failed: {str(e)}")
            print("Model evaluation will proceed without MLflow tracking")
            # Calculate and save metrics locally even if MLflow fails
            predicted_qualities = model.predict(test_x)
            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)
            print(f"Metrics saved locally: {self.config.metric_file_name}")