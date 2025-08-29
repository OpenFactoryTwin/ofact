from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchmetrics.regression import R2Score
from permetrics.regression import RegressionMetric
from tqdm import tqdm

from ofact.helpers import root_path_without_lib_name
from ofact.settings import ROOT_PATH


class EarlyStopping:
    def __init__(self, tolerance_epochs=5, threshold=0):
        """EarlyStopping is used to avoid overfitting.
        :param tolerance_epochs: The number of epochs the value must be under the threshold to stop
        :param threshold: The threshold of the loss that must be undertaken to stop"""

        self.tolerance_epochs = tolerance_epochs
        self.threshold = threshold
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        if loss > self.threshold:
            self.counter = 0
            return

        self.counter += 1
        if self.counter >= self.tolerance_epochs:
            self.early_stop = True


class NeuralNetwork(nn.Module):

    def __init__(self, input_size, output_size,
                 neurons_per_layer=[40, 20, 10],
                 activation_func=nn.LeakyReLU,
                 use_batch_norm=False,
                 dropout_func: Optional[Union[nn.Dropout, nn.AlphaDropout]] = None, dropout_prob=[0.5, 0.5, 0.5],
                 model_path=None,
                 device: str = "cpu"):
        """

        :param input_size:
        :param output_size:
        :param neurons_per_layer:
        :param activation_func:
        :param use_batch_norm: used for faster learning ToDo
        (reached through higher independence from the varying input variance of the last layer)
        Note: should be used for greater batch sizes
        :param dropout_func: used to randomly drop some of the neurons in the training phase with probability p
        to avoid overfitting to the training data
        Alternatives are Dropout or AlphaDropout
        :param dropout_prob: probability of the dropout to that can be set for each layer independently
        dropout should be used with a probability of 0.2 - 0.5 (0.5 is the default)
        :param model_path:
        """

        super(NeuralNetwork, self).__init__()

        if device != "cpu":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[{type(self).__name__:<20s}] Device: {self.device}")

        self.input_size = input_size
        self.output_size = output_size
        self.neurons_per_layer = neurons_per_layer
        self.activation_func = activation_func
        self.use_batch_norm = use_batch_norm
        self.dropout_func = dropout_func
        self.dropout_prob = dropout_prob

        self.model = self._get_neural_network(neurons_per_layer, use_batch_norm, dropout_func, dropout_prob,
                                              activation_func)

        self.model_path = model_path
        if model_path is not None:
            self._load_model(model_path)

        else:
            self.loss_history = []

        self.model.to(self.device)

    def _get_neural_network(self, neurons_per_layer, use_batch_norm, dropout_func, dropout_prob, activation_func):
        in_features = self.input_size
        layers = []
        for idx, neurons in enumerate(neurons_per_layer):
            linear_layer = nn.Linear(in_features, neurons)
            # torch.nn.init.kaiming_normal_(linear_layer.weight, a=0.0003)  # ToDo: should also be an input
            layers.append(linear_layer)

            activation_layer = activation_func()
            if use_batch_norm:
                batch_layer = nn.BatchNorm1d(neurons)
                layers.append(batch_layer)

            layers.append(activation_layer)
            if dropout_func is not None:
                dropout_layer = dropout_func(dropout_prob[idx])
                layers.append(dropout_layer)

            in_features = neurons

        layers.append(nn.Linear(in_features, self.output_size))

        model = nn.Sequential(*layers)
        return model

    def describe_model(self):
        """Deliver a static description of the model."""
        input_output_size = f"Input size: {self.input_size} Output size: {self.output_size}"
        layers = f"Layers: {self.neurons_per_layer} Activation function: {self.activation_func}"
        batch_norm = f"Use batch_norm: {self.use_batch_norm}"
        dropout = f"Dropout: {self.dropout_func} with probability: {self.dropout_prob}"
        static_description = input_output_size + "\n" + layers + "\n" + batch_norm + "\n" + dropout
        return static_description

    def get_init_parameters(self):
        # ToDo: should be an abstract method
        raise NotImplementedError

    def update_device(self, device="cpu"):
        if device != "cpu":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def _load_model(self, model_path):
        """Load the model ..."""
        complete_model_path = Path(root_path_without_lib_name(ROOT_PATH) + model_path)
        self.model.load_state_dict(torch.load(complete_model_path))
        self.model.eval()

    def save_model(self, model_path):
        """Save the model ..."""
        complete_model_path = Path(root_path_without_lib_name(ROOT_PATH) + model_path)
        torch.save(self.model.state_dict(), complete_model_path)

    def perform_training(self, dataloader, loss_fn, optimizer_choice, lr, epochs=100,
                         use_regularization=False, regularization_norm=2, regularization_lambda=0.01,
                         use_early_stopping=False, early_stopping_tolerance_epochs=5, early_stopping_threshold=0):
        """
        Train the neural network model on the given dataset.
        :param dataloader: The dataloader to use for training
        :param loss_fn: The loss function to use
        :param optimizer_choice: The optimizer to use
        :param lr: The learning rate
        :param epochs: The number of epochs to use
        :param use_regularization: Whether to use regularization
        :param regularization_norm: The type of regularization to use
        :param regularization_lambda: The coefficient used to determine the threshold for outliers
        :param use_early_stopping: Whether to use early stopping
        :param early_stopping_tolerance_epochs: The number of epochs the value must be under the threshold to stop
        :param early_stopping_threshold: The threshold of the loss that must be undertaken to stop
        """

        optimizer = optimizer_choice(self.parameters(), lr=lr)

        if use_early_stopping:
            early_stopping = EarlyStopping(tolerance_epochs=early_stopping_tolerance_epochs,
                                           threshold=early_stopping_threshold)

        loss_history = []
        progress_bar = tqdm(range(epochs), colour="green", desc='Training Phase', leave=False, dynamic_ncols=True,
                            smoothing=0, disable=False)
        for t in progress_bar:
            for batch, (X, y) in enumerate(dataloader):
                # Compute prediction and loss
                pred = self(X)
                epoch_train_loss = self._get_loss(pred, y, loss_fn)

                # Regularization
                if use_regularization:
                    all_params = torch.cat([param.view(-1)
                                            for name, param in self.model.named_parameters()
                                            if 'bias' not in name])
                    l_regularization = torch.norm(all_params, p=regularization_norm)
                    epoch_train_loss += regularization_lambda * l_regularization

                # Backpropagation
                optimizer.zero_grad()
                epoch_train_loss.backward()
                optimizer.step()

                epoch_train_loss = epoch_train_loss.item()
                loss_history.append(epoch_train_loss)
                if batch % 25 == 0:
                    progress_bar.set_postfix({"Loss": f"{epoch_train_loss:>7f}"})

            # early stopping
            if use_early_stopping:
                early_stopping(epoch_train_loss)
                if early_stopping.early_stop:
                    print(f"The training stops by early stopping at epoch: {t}")
                    break

        self.loss_history = loss_history

        return loss_history

    def _get_loss(self, pred, y, loss_fn):
        return loss_fn(pred, y)

    def plot_loss_history(self, loss_history=None):
        """Plot the loss_history of the model learning process."""
        # print("Loss History: {}".format(loss_history))
        plt.plot(np.arange(len(loss_history)), loss_history)
        plt.title("loss_history")
        plt.show()

    def get_predicted_value(self, sample):
        """Used in the execution phase"""

        predicted_value = self._predict(sample)
        predicted_value = predicted_value.item()
        # print("Predicted value:", predicted_value, self.__class__.__name__)
        return predicted_value

    def _predict(self, sample):
        """Predict the output of the model. (Used within the training process)"""
        if type(sample) == np.ndarray:
            sample = torch.from_numpy(sample.astype(np.float32)).to(self.device)
            # sample = torch.from_numpy(sample)[:, None].to(self.device)
        self.model.eval()
        predicted_value = self.model(sample)
        return predicted_value


class RegressionNeuralNetwork(NeuralNetwork):

    def __init__(self, input_size, output_size=1,
                 neurons_per_layer=[40, 20, 10],
                 activation_func=nn.LeakyReLU,
                 use_batch_norm=False,
                 dropout_func: Optional[Union[nn.Dropout, nn.AlphaDropout]] = None, dropout_prob=[0.5, 0.5, 0.5],
                 model_path=None,
                 device: str = "cpu"):
        super().__init__(input_size=input_size, output_size=output_size,
                         neurons_per_layer=neurons_per_layer,
                         activation_func=activation_func,
                         use_batch_norm=use_batch_norm,
                         dropout_func=dropout_func, dropout_prob=dropout_prob,
                         model_path=model_path,
                         device=device)

    def forward(self, x):
        logits = self.model(x).squeeze(axis=1)
        return logits

    def check_accuracy(self, loader):
        # ToDo
        #  refactor - performance measure
        #  output should be a dict

        residuals = []
        rmse_loss_list = []
        nrmse_loss_list = []
        number_of_samples = 0
        self.model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                predictions = self(x)
                value = self.r2_score(predictions, y).item()
                mse_loss = nn.MSELoss()
                rmse_loss = torch.sqrt(mse_loss(predictions, y)).detach().cpu().numpy()

                evaluator = RegressionMetric(predictions.detach().cpu().numpy(), y.detach().cpu().numpy(), decimal=5)
                nrmse = evaluator.normalized_root_mean_square_error()

                weight = y.shape[0]
                number_of_samples += weight
                rmse_loss_list.append(rmse_loss * weight)
                nrmse_loss_list.append(nrmse * weight)
                residuals.append(value * weight)

        self.model.train()

        combined_residuals = sum(residuals) / number_of_samples
        rmse_mean = sum(rmse_loss_list) / number_of_samples
        nrmse_mean = sum(nrmse_loss_list) / number_of_samples

        return combined_residuals, None, rmse_mean, nrmse_mean

    def r2_score(self, predictions, y):
        r2score = R2Score()
        r2score.to(self.device)
        results = r2score(predictions, y)

        return results


class NormalDistributionNeuralNetwork(RegressionNeuralNetwork):

    def __init__(self, input_size, output_size=2,
                 neurons_per_layer=[40, 20, 10],
                 activation_func=nn.LeakyReLU,
                 use_batch_norm=False,
                 dropout_func: Optional[Union[nn.Dropout, nn.AlphaDropout]] = None, dropout_prob=[0.5, 0.5, 0.5],
                 model_path=None,
                 device: str = "cpu"):
        super().__init__(input_size=input_size, output_size=output_size,
                         neurons_per_layer=neurons_per_layer,
                         activation_func=activation_func,
                         use_batch_norm=use_batch_norm,
                         dropout_func=dropout_func, dropout_prob=dropout_prob,
                         model_path=model_path, device=device)

    def _get_neural_network(self, neurons_per_layer, use_batch_norm, dropout_func, dropout_prob, activation_func):
        in_features = self.input_size
        layers = []
        for idx, neurons in enumerate(neurons_per_layer):
            linear_layer = nn.Linear(in_features, neurons)
            # torch.nn.init.kaiming_normal_(linear_layer.weight, a=0.0003)  # ToDo: should also be an input
            layers.append(linear_layer)

            if use_batch_norm:
                batch_layer = nn.BatchNorm1d(neurons)
                layers.append(batch_layer)

            activation_layer = activation_func()
            layers.append(activation_layer)
            if dropout_func is not None:
                dropout_layer = dropout_func(dropout_prob[idx])
                layers.append(dropout_layer)

            in_features = neurons

        linear_layer = nn.Linear(in_features, self.output_size)
        layers.append(linear_layer)
        layers.append(nn.ReLU())

        model = nn.Sequential(*layers)
        return model

    def forward(self, x):
        outputs = self.model(x)
        mue_array = torch.max(outputs[:, 0], torch.full(outputs[:, 0].shape, 1e-12).to(device=self.device))
        sigma_array = torch.max(outputs[:, 1], torch.full(outputs[:, 1].shape, 1).to(device=self.device))  # one is minimum for ELU activation
        outputs = torch.distributions.normal.Normal(loc=mue_array, scale=sigma_array)
        return outputs

    def _get_loss(self, pred, y, loss_fn):
        loss = (-1 * pred.log_prob(y)).mean()

        return loss

    def check_accuracy(self, loader):
        residuals = []
        deviations_from_mean = []
        number_of_samples = 0
        self.model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                predictions = self(x)
                r2_results, deviations_probability = self.score(predictions, y)
                weight = y.shape[0]
                number_of_samples += weight
                residuals.append(r2_results * weight)
                deviations_from_mean.append(deviations_probability)

        self.model.train()

        combined_residuals = (sum(residuals) / number_of_samples)

        deviations_from_mean = torch.cat(deviations_from_mean, dim=0).mean()

        return combined_residuals, deviations_from_mean, None, None

    def score(self, predictions, y):
        r2score = R2Score()
        r2score.to(self.device)
        r2_results = r2score(predictions.loc, y).cpu()
        deviations_probability = predictions.cdf(y) - 0.5

        return r2_results, deviations_probability
