
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from typing import Optional
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm

from ofact.helpers import root_path_without_lib_name
from ofact.settings import ROOT_PATH


class BaseEnsembleModel:
    """
    Base class for ensemble models with common functionality.
    """
    def __init__(self, input_size: int, output_size: int = 1, model_path: Optional[str] = None):
        self.input_size = input_size
        self.output_size = output_size
        self.model_path = model_path
        self.model = self._get_model()
        self.model_history = []

        if model_path is not None:
            self._load_model(model_path)

    def _get_model(self):
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_model.")

    def _load_model(self, model_path: str):
        """Load a saved model."""
        complete_model_path = Path(root_path_without_lib_name(ROOT_PATH) + model_path)
        self.model = joblib.load(complete_model_path)

    def save_model(self, model_path: str):
        """Save the model."""
        complete_model_path = Path(root_path_without_lib_name(ROOT_PATH) + model_path)
        joblib.dump(self.model, complete_model_path)

    def perform_training(self, dataloader, epochs: int = 100, use_early_stopping: bool = False,
                         early_stopping_threshold: float = 0, early_stopping_tolerance_epochs: int = 5):
        """
        Train the model on the given dataset.
        """
        X_all, y_all = [], []
        for batch, (X, y) in enumerate(tqdm(dataloader, desc="Preparing training data", leave=False)):
            X_all.append(X.numpy() if hasattr(X, "numpy") else X)
            y_all.append(y.numpy() if hasattr(y, "numpy") else y)

        X_all = np.vstack(X_all)
        y_all = np.vstack(y_all)

        self.model.fit(X_all, y_all)
        return self.model

    def get_predicted_value(self, sample):
        return self._predict(sample)

    def _predict(self, sample):
        """Make a prediction for a sample."""
        if isinstance(sample, np.ndarray):
            sample = sample.astype(np.float32)
        elif hasattr(sample, "numpy"):
            sample = sample.numpy()

        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        return np.atleast_1d(self.model.predict(sample))

    def check_accuracy(self, loader):
        """Calculate metrics for the evaluation dataset."""
        residuals = []
        rmse_list = []
        nrmse_list = []
        number_of_samples = 0

        for x, y in loader:
            x = x.numpy() if hasattr(x, "numpy") else x
            y = np.atleast_1d(y.numpy() if hasattr(y, "numpy") else y)

            predictions = self._predict(x)

            value = r2_score(y, predictions)
            rmse = np.sqrt(mean_squared_error(y, predictions))
            nrmse = self._normalized_root_mean_square_error(y, predictions)

            weight = y.shape[0]
            number_of_samples += weight
            residuals.append(value * weight)
            rmse_list.append(rmse * weight)
            nrmse_list.append(nrmse * weight)

        combined_residuals = sum(residuals) / number_of_samples
        rmse_mean = sum(rmse_list) / number_of_samples
        nrmse_mean = sum(nrmse_list) / number_of_samples

        return {
            "R2": combined_residuals,
            "RMSE": rmse_mean,
            "NRMSE": nrmse_mean
        }

    def _normalized_root_mean_square_error(self, y_true, y_pred):
        """Calculate normalized RMSE."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        y_range = np.max(y_true) - np.min(y_true)
        return rmse / y_range if y_range != 0 else 0


class RandomForestRegression(BaseEnsembleModel):
    """
    Random Forest regression model as a child class of the base class.
    """
    def __init__(self, input_size: int, output_size: int = 1,
                 n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 use_bootstrap: bool = True, model_path: Optional[str] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.use_bootstrap = use_bootstrap

        super().__init__(input_size, output_size, model_path)

    def _get_model(self):
        """Create the Random Forest model."""
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.use_bootstrap
        )
        return model

    def describe_model(self):
        """Provide a description of the model."""
        description = (
            f"Input size: {self.input_size} Output size: {self.output_size}\n"
            f"n_estimators: {self.n_estimators} max_depth: {self.max_depth}\n"
            f"min_samples_split: {self.min_samples_split} min_samples_leaf: {self.min_samples_leaf}\n"
            f"Use bootstrap: {self.use_bootstrap}"
        )
        return description

    def get_init_parameters(self):
        # ToDo: should be an abstract method
        return {"input_size": self.input_size, "output_size": self.output_size,
                "n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split, "min_samples_leaf": self.min_samples_leaf,
                "use_bootstrap": self.use_bootstrap, "model_path": self.model_path}
